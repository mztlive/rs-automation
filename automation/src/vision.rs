use anyhow::Result;
use opencv::{
    core::{self, Mat, Point, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};
use xcap::image::RgbaImage;

/// 模板匹配的结果，包含左上角坐标、匹配分数以及模板尺寸。
pub struct TemplateMatch {
    /// 匹配结果的左上角（像素）。
    pub top_left: Point,
    /// 0.0–1.0 的匹配分数。
    pub score: f64,
    /// 模板宽度（像素）。
    pub tpl_w: i32,
    /// 模板高度（像素）。
    pub tpl_h: i32,
    /// 有效（非透明）区域的中心点，基于模板左上角的偏移（像素）。
    pub center: Option<(f64, f64)>,
}

/// 在 RGBA 截图上寻找模板的位置，自动处理颜色空间转换。
pub fn find_template_pos_in_rgba(image: &RgbaImage, template_path: &str) -> Result<TemplateMatch> {
    let screenshot_bgr = rgba_to_bgr(image)?;
    match_template_on_bgr(&screenshot_bgr, template_path)
}

/// 将 `xcap` 提供的 RGBA 图像复制到 OpenCV `Mat` 并转换为 BGR。
pub fn rgba_to_bgr(image: &RgbaImage) -> Result<Mat> {
    let (w, h) = image.dimensions();
    let mut screenshot_rgba = Mat::zeros(h as i32, w as i32, opencv::core::CV_8UC4)?.to_mat()?;
    {
        let src = image.as_raw();
        let dst = screenshot_rgba.data_bytes_mut()?;
        dst.copy_from_slice(src);
    }
    let mut screenshot_bgr = Mat::default();
    imgproc::cvt_color(
        &screenshot_rgba,
        &mut screenshot_bgr,
        imgproc::COLOR_RGBA2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(screenshot_bgr)
}

/// 在给定的 BGR Mat 上执行模板匹配，必要时利用 alpha 通道生成 mask。
pub fn match_template_on_bgr(screenshot_bgr: &Mat, template_path: &str) -> Result<TemplateMatch> {
    match_template_on_bgr_multi(screenshot_bgr, template_path, &[1.0])
}

/// 将模板匹配的结果可视化：绘制矩形、中心点与分数后写入磁盘。
#[allow(dead_code)]
pub fn save_debug_visualization(
    image: &RgbaImage,
    top_left_px: Point,
    tpl_w: i32,
    tpl_h: i32,
    score: f64,
    out_path: &str,
) -> Result<()> {
    // Convert screenshot to BGR Mat
    let (w, h) = image.dimensions();
    let mut mat_rgba = Mat::zeros(h as i32, w as i32, opencv::core::CV_8UC4)?.to_mat()?;
    {
        let src = image.as_raw();
        let dst = mat_rgba.data_bytes_mut()?;
        dst.copy_from_slice(src);
    }
    let mut mat_bgr = Mat::default();
    imgproc::cvt_color(
        &mat_rgba,
        &mut mat_bgr,
        imgproc::COLOR_RGBA2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Draw rectangle and center point
    imgproc::rectangle(
        &mut mat_bgr,
        core::Rect::new(top_left_px.x, top_left_px.y, tpl_w as i32, tpl_h as i32),
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    let center = Point::new(top_left_px.x + tpl_w / 2, top_left_px.y + tpl_h / 2);
    imgproc::circle(
        &mut mat_bgr,
        center,
        4,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    // Put text for score
    let text = format!("score={:.3}", score);
    imgproc::put_text(
        &mut mat_bgr,
        &text,
        Point::new(top_left_px.x.max(0), (top_left_px.y - 6).max(0)),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    // Save image
    imgcodecs::imwrite(out_path, &mat_bgr, &core::Vector::new())?;
    Ok(())
}

/// 在给定的 BGR Mat 上执行多尺度模板匹配，返回得分最高的命中。
pub fn match_template_on_bgr_multi(
    screenshot_bgr: &Mat,
    template_path: &str,
    scales: &[f64],
) -> Result<TemplateMatch> {
    // Load template with alpha if available
    let template_raw = imgcodecs::imread(template_path, imgcodecs::IMREAD_UNCHANGED)?;
    if template_raw.empty() {
        anyhow::bail!("模板读取失败: {}", template_path);
    }
    let tpl_channels = template_raw.channels();
    let mut template_bgr_base = Mat::default();
    let mut mask_base = Mat::default();
    if tpl_channels == 4 {
        imgproc::cvt_color(
            &template_raw,
            &mut template_bgr_base,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut alpha = Mat::default();
        core::extract_channel(&template_raw, &mut alpha, 3)?; // alpha channel
        imgproc::threshold(&alpha, &mut mask_base, 0.0, 255.0, imgproc::THRESH_BINARY)?;
    } else {
        template_bgr_base = template_raw;
    }

    let center_base = if !mask_base.empty() {
        mask_center(&mask_base)?
    } else {
        None
    };

    let mut best: Option<TemplateMatch> = None;
    for scale in scales.iter().copied().filter(|s| *s > 0.0) {
        let mut template_bgr = Mat::default();
        let mut mask = Mat::default();

        if (scale - 1.0).abs() < f64::EPSILON {
            template_bgr = template_bgr_base.try_clone()?;
            mask = mask_base.try_clone()?;
        } else {
            let new_size = core::Size {
                width: (template_bgr_base.cols() as f64 * scale).round() as i32,
                height: (template_bgr_base.rows() as f64 * scale).round() as i32,
            };
            if new_size.width <= 1 || new_size.height <= 1 {
                continue;
            }
            imgproc::resize(
                &template_bgr_base,
                &mut template_bgr,
                new_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            if !mask_base.empty() {
                let mut scaled_mask = Mat::default();
                imgproc::resize(
                    &mask_base,
                    &mut scaled_mask,
                    new_size,
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;
                let mut bin_mask = Mat::default();
                imgproc::threshold(
                    &scaled_mask,
                    &mut bin_mask,
                    0.0,
                    255.0,
                    imgproc::THRESH_BINARY,
                )?;
                mask = bin_mask;
            }
        }

        let result_cols = screenshot_bgr.cols() - template_bgr.cols() + 1;
        let result_rows = screenshot_bgr.rows() - template_bgr.rows() + 1;
        if result_cols <= 0 || result_rows <= 0 {
            continue; // template larger than screenshot at this scale
        }

        let mut result = Mat::zeros(result_rows, result_cols, opencv::core::CV_32FC1)?.to_mat()?;
        if !mask.empty() {
            imgproc::match_template(
                &screenshot_bgr,
                &template_bgr,
                &mut result,
                imgproc::TM_CCORR_NORMED,
                &mask,
            )?;
        } else {
            imgproc::match_template(
                &screenshot_bgr,
                &template_bgr,
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &core::no_array(),
            )?;
        }

        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = Point::new(0, 0);
        let mut max_loc = Point::new(0, 0);
        core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &core::no_array(),
        )?;

        if best.as_ref().map(|b| max_val > b.score).unwrap_or(true) {
            best = Some(TemplateMatch {
                top_left: max_loc,
                score: max_val,
                tpl_w: template_bgr.cols(),
                tpl_h: template_bgr.rows(),
                center: center_base.map(|(cx, cy)| (cx * scale, cy * scale)),
            });
        }
    }

    best.ok_or_else(|| anyhow::anyhow!("模板匹配失败（所有尺度均未命中）"))
}

/// 计算二值 mask 的非零区域中心点（像素）。返回 None 表示 mask 为空。
fn mask_center(mask: &Mat) -> Result<Option<(f64, f64)>> {
    let rows = mask.rows();
    let cols = mask.cols();
    if rows <= 0 || cols <= 0 {
        return Ok(None);
    }

    let mut min_x = cols;
    let mut min_y = rows;
    let mut max_x = -1;
    let mut max_y = -1;

    for y in 0..rows {
        for x in 0..cols {
            let v = *mask.at_2d::<u8>(y, x)?;
            if v != 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if max_x < 0 || max_y < 0 {
        return Ok(None);
    }

    let cx = (min_x as f64 + max_x as f64) / 2.0;
    let cy = (min_y as f64 + max_y as f64) / 2.0;
    Ok(Some((cx, cy)))
}
