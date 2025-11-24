use super::Step;
use crate::{input, page, pipeline::RunCtx, vision, window::WandaWindow};
use anyhow::Result;

/// 根据 page.json 中定义的锚点模板执行点击。
///
/// - `page`: 页面名称，需与 ML 输出及配置文件保持一致。
/// - `anchor`: 锚点名称，用于查找模板图片与阈值。
pub struct ClickAnchor {
    pub page: &'static str,
    pub anchor: &'static str,
    pub pos: AnchorClickPos,
}

#[derive(Clone, Copy)]
pub enum AnchorClickPos {
    Center,
    Left,
    Right,
    Top,
    Bottom,
    /// 自定义点击位置，基于模板左上角的相对比例（0.0–1.0）。
    Relative {
        x: f32,
        y: f32,
    },
}

impl Default for AnchorClickPos {
    fn default() -> Self {
        AnchorClickPos::Center
    }
}

impl AnchorClickPos {
    fn target_px(&self, m: &vision::TemplateMatch) -> (f64, f64) {
        match self {
            AnchorClickPos::Center => {
                if let Some((cx, cy)) = m.center {
                    (m.top_left.x as f64 + cx, m.top_left.y as f64 + cy)
                } else {
                    (
                        m.top_left.x as f64 + m.tpl_w as f64 * 0.5,
                        m.top_left.y as f64 + m.tpl_h as f64 * 0.5,
                    )
                }
            }
            AnchorClickPos::Left => Self::ratio_point(m, 0.25, 0.5),
            AnchorClickPos::Right => Self::ratio_point(m, 0.75, 0.5),
            AnchorClickPos::Top => Self::ratio_point(m, 0.5, 0.25),
            AnchorClickPos::Bottom => Self::ratio_point(m, 0.5, 0.75),
            AnchorClickPos::Relative { x, y } => Self::ratio_point(m, *x, *y),
        }
    }

    fn ratio_point(m: &vision::TemplateMatch, x: f32, y: f32) -> (f64, f64) {
        let clamped_x = x.clamp(0.0, 1.0);
        let clamped_y = y.clamp(0.0, 1.0);
        (
            m.top_left.x as f64 + m.tpl_w as f64 * clamped_x as f64,
            m.top_left.y as f64 + m.tpl_h as f64 * clamped_y as f64,
        )
    }
}

impl Step for ClickAnchor {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let anchor = page::anchor(self.page, self.anchor)?;
        let image = ctx.capture_rgba(window)?;
        let (w, h) = window.size()?;
        let (x, y) = window.position()?;

        let m = vision::find_template_pos_in_rgba(image, anchor.image.as_str())?;
        let (img_w, img_h) = image.dimensions();
        let scale_x = img_w as f32 / w as f32;
        let scale_y = img_h as f32 / h as f32;

        let (target_px_x, target_px_y) = self.pos.target_px(&m);
        let logical_x = target_px_x as f32 / scale_x;
        let logical_y = target_px_y as f32 / scale_y;
        let sx = x + logical_x.round() as i32;
        let sy = y + logical_y.round() as i32;

        if m.score >= anchor.threshold {
            input::click_screen(sx, sy)?;
            ctx.invalidate();
        }

        Ok(())
    }
}
