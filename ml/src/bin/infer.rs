#![recursion_limit = "256"]

use anyhow::{Context, Result};
use burn::backend::wgpu::{self, RuntimeOptions, graphics::Metal};
use burn::module::Module;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::{Tensor, TensorData, activation::softmax, backend::Backend};
use clap::Parser;
use image::DynamicImage;
use ml::model::PageNet;
use std::{fs::File, path::PathBuf};

type InferenceBackend = wgpu::Metal<f32>;

#[derive(Parser, Debug)]
#[command(name = "ml-infer", about = "加载训练好的页面分类模型并进行推理")]
struct Args {
    /// 模型所在目录（包含 page_net 和 labels.json）
    #[arg(long)]
    model_dir: PathBuf,

    /// 待识别的图像路径
    #[arg(long)]
    image: PathBuf,

    /// 模型训练时使用的输入宽度
    #[arg(long, default_value_t = 256)]
    width: u32,

    /// 模型训练时使用的输入高度
    #[arg(long, default_value_t = 480)]
    height: u32,

    /// 是否对像素做 [0,1] 归一化
    #[arg(long, default_value_t = true)]
    normalize: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = <InferenceBackend as Backend>::Device::default();
    let _ = wgpu::init_setup::<Metal>(&device, RuntimeOptions::default());

    let labels = load_labels(args.model_dir.join("labels.json"))?;

    let mut model = PageNet::<InferenceBackend>::new(
        &device,
        args.height as usize,
        args.width as usize,
        labels.len(),
    );

    model = model
        .load_file(
            args.model_dir.join("page_net"),
            &DefaultFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .context("加载模型权重失败")?;

    let image = image::open(&args.image)
        .with_context(|| format!("无法读取图像 {}", args.image.display()))?;
    let tensor = preprocess(&image, args.width, args.height, args.normalize, &device);

    let logits = model.forward(tensor);
    let probs = softmax(logits.clone(), 1);
    let preds = logits.argmax(1);

    let class_index = preds
        .into_data()
        .into_vec::<i32>()
        .map_err(|err| anyhow::anyhow!("读取预测结果失败：{err:?}"))?[0]
        as usize;

    let probabilities = probs
        .into_data()
        .into_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("读取概率结果失败：{err:?}"))?;

    println!(
        "预测结果：{} (#{})\n",
        labels.get(class_index).unwrap_or(&"<未知>".to_string()),
        class_index
    );

    for (idx, label) in labels.iter().enumerate() {
        let prob = probabilities[idx];
        println!("  {:>3}: {:<20} {:.4}", idx, label, prob);
    }

    Ok(())
}

fn preprocess<B: Backend>(
    image: &DynamicImage,
    width: u32,
    height: u32,
    normalize: bool,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut resized = image.to_rgb8();
    if resized.width() != width || resized.height() != height {
        resized = image::imageops::resize(
            &resized,
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        );
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut data = vec![0.0f32; 3 * width_usize * height_usize];

    for y in 0..height_usize {
        for x in 0..width_usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let value = pixel[c] as f32;
                let value = if normalize { value / 255.0 } else { value };
                let idx = c * width_usize * height_usize + y * width_usize + x;
                data[idx] = value;
            }
        }
    }

    Tensor::<B, 4>::from_data(
        TensorData::new(data, [1, 3, height_usize, width_usize]),
        device,
    )
}

fn load_labels(path: PathBuf) -> Result<Vec<String>> {
    let file = File::open(&path).with_context(|| format!("无法读取标签文件 {}", path.display()))?;
    let labels: Vec<String> = serde_json::from_reader(file)
        .with_context(|| format!("解析标签文件失败 {}", path.display()))?;
    Ok(labels)
}
