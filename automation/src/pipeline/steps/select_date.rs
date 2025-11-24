use anyhow::{Result, anyhow};
use image::GenericImageView;
use ocr::RecognizedText;
use std::collections::HashMap;

use crate::pipeline::RunCtx;
use crate::pipeline::steps::primitive::condition::{build_ocr_engine, to_dynamic_image};
use crate::pipeline::steps::primitive::{Condition, HorizontalScroll, LoopUntil, Sequence};
use crate::window::WandaWindow;
use crate::{input, pipeline::Step};

/// 通过 OCR 找到日期行，将鼠标移入后做水平滚动，直到命中目标日期并点击。
///
/// 内部组合已有原语：`Sequence` + `LoopUntil` + `HorizontalScroll` + `ClickOcrMatch`。
pub struct SelectDateByOcr {
    pub target_pattern: String,
    pub max_scroll_attempts: u32,
    pub scroll_pixels: i32,
}

impl Step for SelectDateByOcr {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        if self.max_scroll_attempts == 0 {
            return Err(anyhow!("max_scroll_attempts must be > 0"));
        }

        let sequence = Sequence::new()
            .step(MoveMouseToDateRow)
            .step(LoopUntil {
                label: "find-date",
                cond: Condition::TextContains {
                    pattern: self.target_pattern.clone(),
                    case_sensitive: false,
                },
                on_miss: HorizontalScroll {
                    lines: self.scroll_pixels,
                    settle_ms: 150,
                },
                max_iters: self.max_scroll_attempts as usize,
                delay_ms: Some(120),
            })
            .step(crate::pipeline::steps::primitive::ClickOcrMatch {
                patterns: self.target_pattern.clone(),
                case_sensitive: false,
            });

        sequence.run(window, ctx)
    }

    fn label(&self) -> &'static str {
        "SelectDateByOcr"
    }
}

/// 找到包含“月/日”或“今天/明天/后天”等文本的行，并把鼠标移动到该行的中点。
struct MoveMouseToDateRow;

impl Step for MoveMouseToDateRow {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let (recognized, dims) = ensure_ocr(window, ctx)?;

        let Some(row_bounds) = find_date_row_bounds(&recognized) else {
            return Err(anyhow!("未找到日期文本行，无法移动鼠标"));
        };

        let window_pos = window.position()?;
        let window_size = window.size()?;

        let row_center = (
            (row_bounds.0 + row_bounds.1) / 2,
            (row_bounds.2 + row_bounds.3) / 2,
        );
        let screen_point = to_screen_point(
            row_center,
            dims,
            window_pos,
            (window_size.0 as i32, window_size.1 as i32),
        );
        input::move_mouse(screen_point.0, screen_point.1)?;
        Ok(())
    }

    fn label(&self) -> &'static str {
        "MoveMouseToDateRow"
    }
}

fn ensure_ocr(
    window: &mut WandaWindow,
    ctx: &mut RunCtx,
) -> Result<(Vec<RecognizedText>, (u32, u32))> {
    let rgba = ctx.capture_rgba(window)?;
    let dyn_img = to_dynamic_image(rgba)?;
    let engine = ctx.ensure_ocr_engine(build_ocr_engine)?;
    let results = engine.recognize_image(&dyn_img)?;
    let dims = dyn_img.dimensions();
    ctx.set_ocr_results(results.clone(), dims);
    Ok((results, dims))
}

/// 返回最密集日期文本行的包围盒：(min_x, max_x, min_y, max_y)。
fn find_date_row_bounds(recognized: &[RecognizedText]) -> Option<(u32, u32, u32, u32)> {
    let bucket = 15; // y 方向聚类粒度
    let mut rows: HashMap<i32, Vec<&RecognizedText>> = HashMap::new();

    println!(
        "RecognizedTexts: {:?}",
        recognized
            .iter()
            .map(|r| r.text.clone())
            .collect::<Vec<_>>()
    );

    for r in recognized.iter().filter(|r| is_date_like(&r.text)) {
        println!("text: {}", r.text);
        let center_y = (r.bbox.y + r.bbox.height / 2) as i32;
        let key = center_y / bucket;
        rows.entry(key).or_default().push(r);
    }

    let (_key, row) = rows.into_iter().max_by_key(|(_, v)| v.len())?;

    let min_x = row.iter().map(|r| r.bbox.x).min().unwrap_or(0);
    let max_x = row
        .iter()
        .map(|r| r.bbox.x + r.bbox.width)
        .max()
        .unwrap_or(0);
    let min_y = row.iter().map(|r| r.bbox.y).min().unwrap_or(0);
    let max_y = row
        .iter()
        .map(|r| r.bbox.y + r.bbox.height)
        .max()
        .unwrap_or(0);

    Some((min_x, max_x, min_y, max_y))
}

fn is_date_like(text: &str) -> bool {
    let t = text.trim();
    t.contains('月')
        || t.contains('日')
        || t.contains("今天")
        || t.contains("明天")
        || t.contains("后天")
}

fn to_screen_point(
    point: (u32, u32),
    dims: (u32, u32),
    window_pos: (i32, i32),
    window_size: (i32, i32),
) -> (i32, i32) {
    let scale_x = window_size.0 as f32 / dims.0.max(1) as f32;
    let scale_y = window_size.1 as f32 / dims.1.max(1) as f32;
    let screen_x = window_pos.0 + (point.0 as f32 * scale_x).round() as i32;
    let screen_y = window_pos.1 + (point.1 as f32 * scale_y).round() as i32;
    (screen_x, screen_y)
}
