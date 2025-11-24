use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use image_proc::{
    BorderOptions, add_gaussian_noise, adjust_color, estimate_edge_color, gaussian_blur,
    occlude_rect, rotate, scale_preserve, translate,
};
use opencv::core::{BORDER_CONSTANT, Mat, Rect, Scalar, Vector};
use opencv::imgcodecs::{self, IMREAD_COLOR};
use opencv::imgproc;
use opencv::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use walkdir::WalkDir;

/// 标签来源枚举，决定如何为图像分配类别。
#[derive(Copy, Clone, Debug, ValueEnum)]
enum LabelSource {
    /// 使用输入目录的首层子目录名作为标签。
    ParentDir,
    /// 为所有图像使用同一个固定标签。
    Fixed,
}

/// 命令行参数：控制输入输出目录、增强数量与随机性。
#[derive(Parser, Debug)]
#[command(
    name = "image-pre-cli",
    version,
    about = "批量生成训练用图像数据增强样本的工具"
)]
struct Args {
    /// 输入目录，应包含原始截图（可按类别分子目录）
    #[arg(long)]
    input: PathBuf,

    /// 输出目录，将按输入的子目录结构写出增强后的图像
    #[arg(long)]
    output: PathBuf,

    /// 每个输入图像生成的增强样本数
    #[arg(long, default_value_t = 32)]
    variants: u32,

    /// 指定随机种子以复现增强结果
    #[arg(long)]
    seed: Option<u64>,

    /// 是否将原始图像一并复制到输出目录
    #[arg(long, default_value_t = false)]
    keep_original: bool,

    /// 标签生成策略：parent-dir 或 fixed
    #[arg(long, value_enum)]
    label_source: Option<LabelSource>,

    /// 当 label_source=fixed 时使用的标签名
    #[arg(long, requires = "label_source")]
    label_value: Option<String>,

    /// 将增强结果写入的清单文件（CSV，包含 image,label）
    #[arg(long)]
    manifest: Option<PathBuf>,
}

/// 程序入口：解析参数并触发批量增强流程。
fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

/// 遍历输入目录，对每张截图执行增强并写入目标文件夹。
///
/// - 保持原始子目录结构，增强文件按 `原名_aug###.png` 命名。
/// - 可选复制原始截图，便于后续对比。
/// - 支持通过随机种子复现增强结果。
fn run(args: Args) -> Result<()> {
    anyhow::ensure!(
        args.input.is_dir(),
        "输入路径必须是有效目录：{}",
        args.input.display()
    );
    fs::create_dir_all(&args.output)
        .with_context(|| format!("无法创建输出目录 {}", args.output.display()))?;

    let mut rng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(rand::thread_rng()).context("无法初始化随机数生成器（StdRng）")?,
    };

    if args.manifest.is_some() && args.label_source.is_none() {
        anyhow::bail!("生成清单文件需要指定 label_source");
    }

    let mut processed_files = 0usize;
    let mut generated_images = 0usize;
    let mut manifest: Vec<(PathBuf, String)> = Vec::new();

    for entry in WalkDir::new(&args.input)
        .into_iter()
        .filter_map(|res| res.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        if !is_supported_image(path) {
            continue;
        }

        let relative_path = path
            .strip_prefix(&args.input)
            .with_context(|| format!("无法计算相对路径：{}", path.display()))?;
        let parent_relative = relative_path.parent().unwrap_or_else(|| Path::new(""));
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("文件名需为有效的 UTF-8 字符串")?;

        let path_str = path.to_string_lossy();
        let image = imgcodecs::imread(&path_str, IMREAD_COLOR)
            .with_context(|| format!("载入图像失败：{}", path.display()))?;

        processed_files += 1;

        let label = match args.label_source {
            Some(LabelSource::ParentDir) => {
                let mut components = parent_relative.components();
                let first = components
                    .next()
                    .and_then(|comp| comp.as_os_str().to_str())
                    .map(|s| s.to_string());
                first.ok_or_else(|| {
                    anyhow::anyhow!(
                        "文件 {} 缺少上级目录，无法通过 parent-dir 生成标签",
                        path.display()
                    )
                })?
            }
            Some(LabelSource::Fixed) => args
                .label_value
                .clone()
                .expect("clap 已保证 fixed 策略时 label_value 存在"),
            None => String::new(),
        };

        if args.keep_original {
            let dest_dir = args.output.join(parent_relative);
            fs::create_dir_all(&dest_dir)
                .with_context(|| format!("无法创建目录 {}", dest_dir.display()))?;
            let original_path = dest_dir.join(format!("{}.png", file_stem));
            write_image(&image, &original_path)
                .with_context(|| format!("写入原始图像失败：{}", original_path.display()))?;
            if args.label_source.is_some() {
                let relative_output = parent_relative.join(format!("{}.png", file_stem));
                manifest.push((relative_output, label.clone()));
            }
        }

        for idx in 0..args.variants {
            let augmented = augment_image(&image, &mut rng)
                .with_context(|| format!("增强 {} 失败", path.display()))?;
            let dest_dir = args.output.join(parent_relative);
            fs::create_dir_all(&dest_dir)
                .with_context(|| format!("无法创建目录 {}", dest_dir.display()))?;
            let output_path = dest_dir.join(format!("{file_stem}_aug{idx:03}.png"));
            write_image(&augmented, &output_path)
                .with_context(|| format!("写入增强图像失败：{}", output_path.display()))?;
            generated_images += 1;
            if args.label_source.is_some() {
                let relative_output = parent_relative.join(format!("{file_stem}_aug{idx:03}.png"));
                manifest.push((relative_output, label.clone()));
            }
        }
    }

    if let Some(manifest_path) = args.manifest {
        write_manifest(&manifest, &manifest_path)?;
        println!(
            "标签清单已生成：{}（{} 条记录）",
            manifest_path.display(),
            manifest.len()
        );
    }

    println!(
        "完成增强：处理 {} 个文件，生成 {} 张增强图像，输出目录：{}",
        processed_files,
        generated_images,
        args.output.display()
    );

    Ok(())
}

/// 过滤文件扩展名，仅允许 PNG/JPG/JPEG。
fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_ascii_lowercase()),
        Some(ext) if ext == "png" || ext == "jpg" || ext == "jpeg"
    )
}

/// 将 `Mat` 写入磁盘，默认保存为 PNG。
fn write_image(image: &Mat, path: &Path) -> Result<()> {
    let params = Vector::<i32>::new();
    let path_str = path.to_string_lossy();
    let success = imgcodecs::imwrite(&path_str, image, &params)
        .with_context(|| format!("无法保存图像 {}", path.display()))?;
    anyhow::ensure!(success, "OpenCV 写入图像失败：{}", path.display());
    Ok(())
}

/// 输出 image,label CSV，路径按输出目录的相对路径记录。
fn write_manifest(entries: &[(PathBuf, String)], manifest_path: &Path) -> Result<()> {
    if entries.is_empty() {
        anyhow::bail!("未收集到任何标签，清单文件未生成");
    }
    if let Some(parent) = manifest_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("无法创建清单目录 {}", parent.display()))?;
        }
    }
    let mut wtr = csv::Writer::from_path(manifest_path)
        .with_context(|| format!("打开清单文件失败：{}", manifest_path.display()))?;
    wtr.write_record(["image", "label"])?;
    for (path, label) in entries {
        let rel_str = path.to_string_lossy().replace('\\', "/");
        wtr.write_record([rel_str.as_str(), label])?;
    }
    wtr.flush()?;
    Ok(())
}
/// 为单张截图叠加随机扰动，生成增强样本。
///
/// 包含平移、缩放、旋转、色彩抖动、加噪、模糊与遮挡等常见扰动。
fn augment_image(image: &Mat, rng: &mut StdRng) -> Result<Mat> {
    let mut working = image.try_clone()?;
    let edge_color = estimate_edge_color(image)?;

    let constant_border = BorderOptions {
        mode: BORDER_CONSTANT,
        value: edge_color,
    };
    let replicate_border = BorderOptions::default();

    if rng.gen_bool(0.7) {
        let shift_x = rng.gen_range(-3.0..=3.0);
        let shift_y = rng.gen_range(-3.0..=3.0);
        working = translate(&working, shift_x, shift_y, constant_border)?;
    }

    if rng.gen_bool(0.6) {
        let scale = rng.gen_range(0.97..=1.03);
        working = scale_preserve(&working, scale, imgproc::INTER_CUBIC, constant_border)?;
    }

    if rng.gen_bool(0.6) {
        let angle = rng.gen_range(-2.0..=2.0);
        working = rotate(&working, angle, replicate_border)?;
    }

    if rng.gen_bool(0.7) {
        let alpha = rng.gen_range(0.95..=1.05);
        let beta = rng.gen_range(-12.0..=12.0);
        let saturation = rng.gen_range(0.9..=1.1);
        working = adjust_color(&working, alpha, beta, saturation)?;
    }

    if rng.gen_bool(0.5) {
        let stddev = rng.gen_range(4.0..=10.0);
        working = add_gaussian_noise(&working, 0.0, stddev)?;
    }

    if rng.gen_bool(0.35) {
        let sigma = rng.gen_range(0.35..=1.0);
        working = gaussian_blur(&working, sigma, replicate_border)?;
    }

    if rng.gen_bool(0.3) {
        working = apply_occlusion(&working, rng, edge_color)?;
    }

    Ok(working)
}

/// 在图像上随机绘制矩形遮挡，模拟通知弹窗或指针遮挡。
fn apply_occlusion(image: &Mat, rng: &mut StdRng, edge_color: Scalar) -> Result<Mat> {
    let size = image.size()?;
    let width = size.width.max(1);
    let height = size.height.max(1);

    let mut result = image.try_clone()?;
    let occlusion_passes = if rng.gen_bool(0.45) {
        rng.gen_range(2..=4)
    } else {
        1
    };

    for pass in 0..occlusion_passes {
        let (min_ratio_w, max_ratio_w) = if pass == 0 { (0.22, 0.6) } else { (0.08, 0.32) };
        let (min_ratio_h, max_ratio_h) = if pass == 0 {
            (0.20, 0.58)
        } else {
            (0.08, 0.30)
        };

        let occ_width = (rng
            .gen_range(
                (width as f64 * min_ratio_w).max(1.0)..=(width as f64 * max_ratio_w).max(1.0),
            )
            .round() as i32)
            .clamp(1, width);
        let occ_height = (rng
            .gen_range(
                (height as f64 * min_ratio_h).max(1.0)..=(height as f64 * max_ratio_h).max(1.0),
            )
            .round() as i32)
            .clamp(1, height);

        let max_x = width.saturating_sub(occ_width);
        let max_y = height.saturating_sub(occ_height);

        let preferred_center_x = width / 2 - occ_width / 2;
        let preferred_center_y = if rng.gen_bool(0.6) {
            height / 2 - occ_height / 2
        } else {
            (height / 4).saturating_sub(occ_height / 2)
        };

        let jitter_x = (max_x as f64 * rng.gen_range(-0.18..=0.18)).round() as i32;
        let jitter_y = (max_y as f64 * rng.gen_range(-0.18..=0.18)).round() as i32;

        let mut pos_x = preferred_center_x + jitter_x;
        let mut pos_y = preferred_center_y + jitter_y;
        if max_x > 0 {
            pos_x = pos_x.clamp(0, max_x);
        } else {
            pos_x = 0;
        }
        if max_y > 0 {
            pos_y = pos_y.clamp(0, max_y);
        } else {
            pos_y = 0;
        }

        let tint = [
            (edge_color[0] + rng.gen_range(-30.0..=120.0)).clamp(0.0, 255.0),
            (edge_color[1] + rng.gen_range(-30.0..=120.0)).clamp(0.0, 255.0),
            (edge_color[2] + rng.gen_range(-30.0..=120.0)).clamp(0.0, 255.0),
        ];
        let occlusion_color = Scalar::new(tint[0], tint[1], tint[2], 255.0);

        let rect = Rect::new(pos_x, pos_y, occ_width, occ_height);
        result = occlude_rect(&result, rect, occlusion_color)?;

        if rng.gen_bool(0.5) {
            let border_color = Scalar::new(
                (tint[0] + rng.gen_range(-40.0..=40.0)).clamp(0.0, 255.0),
                (tint[1] + rng.gen_range(-40.0..=40.0)).clamp(0.0, 255.0),
                (tint[2] + rng.gen_range(-40.0..=40.0)).clamp(0.0, 255.0),
                255.0,
            );
            let thickness = rng.gen_range(2..=4);
            imgproc::rectangle(
                &mut result,
                rect,
                border_color,
                thickness,
                imgproc::LINE_8,
                0,
            )?;
        }

        if pass == 0 && rng.gen_bool(0.55) {
            let button_size =
                (occ_width.min(occ_height) as f64 * rng.gen_range(0.1..=0.18)).round() as i32;
            let button_size = button_size.clamp(6, occ_width.min(occ_height));
            let margin = (button_size as f64 * rng.gen_range(0.3..=0.6)).round() as i32;
            let btn_x = (pos_x + occ_width - button_size - margin)
                .clamp(pos_x, pos_x + occ_width - button_size);
            let btn_y = (pos_y + margin).clamp(pos_y, pos_y + occ_height - button_size);

            let button_color = Scalar::new(
                rng.gen_range(200.0..=255.0),
                rng.gen_range(200.0..=255.0),
                rng.gen_range(200.0..=255.0),
                255.0,
            );
            let button_rect = Rect::new(btn_x, btn_y, button_size, button_size);
            result = occlude_rect(&result, button_rect, button_color)?;

            if rng.gen_bool(0.7) {
                let cross_color = Scalar::new(30.0, 30.0, 30.0, 255.0);
                imgproc::line(
                    &mut result,
                    opencv::core::Point::new(btn_x, btn_y),
                    opencv::core::Point::new(btn_x + button_size, btn_y + button_size),
                    cross_color,
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
                imgproc::line(
                    &mut result,
                    opencv::core::Point::new(btn_x + button_size, btn_y),
                    opencv::core::Point::new(btn_x, btn_y + button_size),
                    cross_color,
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }
    }

    Ok(result)
}
