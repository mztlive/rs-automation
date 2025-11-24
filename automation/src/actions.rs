use anyhow::{Result, anyhow};
use opencv::{core, prelude::*};
use xcap::image::RgbaImage;

use crate::vision;

/// 匹配模板并返回其在屏幕坐标中的中心点以及置信度。
///
/// 将窗口截图转换为 BGR 矩阵后执行模板匹配，然后依据窗口的逻辑尺寸与屏幕位置换算出屏幕像素坐标。
/// 返回的坐标可直接用于 `input::click_screen` 等输入模拟函数。
pub fn find_template_screen_center(
    image: &RgbaImage,
    template_path: &str,
    window_size: (u32, u32),
    window_pos: (i32, i32),
) -> Result<((i32, i32), f64)> {
    let m = vision::find_template_pos_in_rgba(image, template_path)?;

    let (img_w, img_h) = image.dimensions();
    let (win_w, win_h) = window_size;
    let (win_x, win_y) = window_pos;

    // Compute scale from logical window size to physical capture
    let scale_x = img_w as f32 / win_w as f32;
    let scale_y = img_h as f32 / win_h as f32;

    // Center in physical pixels -> logical window coords
    let (center_px_x, center_px_y) = match m.center {
        Some((cx, cy)) => (m.top_left.x as f64 + cx, m.top_left.y as f64 + cy),
        None => (
            m.top_left.x as f64 + m.tpl_w as f64 / 2.0,
            m.top_left.y as f64 + m.tpl_h as f64 / 2.0,
        ),
    };

    let logical_center_x = (center_px_x as f32) / scale_x;
    let logical_center_y = (center_px_y as f32) / scale_y;

    // Convert to absolute screen coords
    let screen_center_x = win_x + logical_center_x.round() as i32;
    let screen_center_y = win_y + logical_center_y.round() as i32;

    Ok(((screen_center_x, screen_center_y), m.score))
}

/// 仅在指定 ROI 内进行模板匹配，并返回匹配中心的屏幕坐标与分数。
/// `roi` 为 `(x, y, w, h)`，基于截图像素坐标。
pub fn find_template_screen_center_in_roi(
    image: &RgbaImage,
    template_path: &str,
    window_size: (u32, u32),
    window_pos: (i32, i32),
    roi: (u32, u32, u32, u32),
) -> Result<((i32, i32), f64)> {
    find_template_screen_center_in_roi_scales(
        image,
        template_path,
        window_size,
        window_pos,
        roi,
        &[1.0],
    )
}

/// 仅在指定 ROI 内进行多尺度模板匹配，并返回匹配中心的屏幕坐标与分数。
/// `scales` 为模板缩放列表（如 `[1.0, 0.9, 0.8, 1.1]`）。
pub fn find_template_screen_center_in_roi_scales(
    image: &RgbaImage,
    template_path: &str,
    window_size: (u32, u32),
    window_pos: (i32, i32),
    roi: (u32, u32, u32, u32),
    scales: &[f64],
) -> Result<((i32, i32), f64)> {
    let (img_w, img_h) = image.dimensions();
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if roi_x + roi_w > img_w || roi_y + roi_h > img_h {
        return Err(anyhow!("ROI 超出截图范围"));
    }

    let screenshot_bgr = vision::rgba_to_bgr(image)?;
    let rect = core::Rect::new(roi_x as i32, roi_y as i32, roi_w as i32, roi_h as i32);
    let roi_mat = screenshot_bgr.roi(rect)?.try_clone()?;
    let m = vision::match_template_on_bgr_multi(&roi_mat, template_path, scales)?;

    let (win_w, win_h) = window_size;
    let (win_x, win_y) = window_pos;
    let scale_x = img_w as f32 / win_w as f32;
    let scale_y = img_h as f32 / win_h as f32;

    // ROI 原点 + 匹配中心 -> 物理像素坐标
    let (center_px_x, center_px_y) = match m.center {
        Some((cx, cy)) => (
            roi_x as f64 + m.top_left.x as f64 + cx,
            roi_y as f64 + m.top_left.y as f64 + cy,
        ),
        None => (
            roi_x as f64 + m.top_left.x as f64 + m.tpl_w as f64 / 2.0,
            roi_y as f64 + m.top_left.y as f64 + m.tpl_h as f64 / 2.0,
        ),
    };

    // 物理像素 -> 逻辑窗口 -> 屏幕
    let logical_center_x = center_px_x as f32 / scale_x;
    let logical_center_y = center_px_y as f32 / scale_y;
    let screen_center_x = win_x + logical_center_x.round() as i32;
    let screen_center_y = win_y + logical_center_y.round() as i32;

    Ok(((screen_center_x, screen_center_y), m.score))
}
