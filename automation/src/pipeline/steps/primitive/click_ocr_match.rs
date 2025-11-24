use super::Step;
use crate::input;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::{Result, anyhow};
use ocr::RecognizedText;

/// 执行 OCR 匹配并点击第一个命中的文本区域中心（屏幕坐标）。
pub struct ClickOcrMatch {
    pub patterns: String,
    pub case_sensitive: bool,
}

impl Step for ClickOcrMatch {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        // 必须依赖已有的 OCR 结果（通常由 Condition::TextContains 产生），避免隐式重新识别。
        let recognized = ctx
            .ocr_results()
            .map(|r| r.to_vec())
            .ok_or_else(|| anyhow!("no cached OCR results; run Condition::TextContains first"))?;
        let dims = ctx
            .ocr_dims()
            .ok_or_else(|| anyhow!("missing OCR dimensions for coordinate conversion"))?;

        let win_pos = window.position()?;
        let (win_w, win_h) = window.size()?;
        let hit = recognized
            .iter()
            .find(|r| has_match(r, &self.patterns, self.case_sensitive))
            .ok_or_else(|| anyhow!("no OCR match for {:?}", self.patterns))?;

        let (cx, cy) = bbox_center(hit);
        let scale_x = win_w as f32 / dims.0.max(1) as f32;
        let scale_y = win_h as f32 / dims.1.max(1) as f32;
        let screen_x = win_pos.0 + (cx as f32 * scale_x).round() as i32;
        let screen_y = win_pos.1 + (cy as f32 * scale_y).round() as i32;

        input::move_mouse(screen_x, screen_y)?;
        input::click_screen(screen_x, screen_y)?;
        ctx.invalidate();
        Ok(())
    }

    fn label(&self) -> &'static str {
        "ClickOcrMatch"
    }
}

fn has_match(result: &RecognizedText, pattern: &str, case_sensitive: bool) -> bool {
    super::text_match::has_similar(&result.text, pattern, case_sensitive)
}

fn bbox_center(result: &RecognizedText) -> (u32, u32) {
    let x = result.bbox.x + result.bbox.width / 2;
    let y = result.bbox.y + result.bbox.height / 2;
    (x, y)
}
