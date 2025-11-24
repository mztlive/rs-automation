use super::Step;
use crate::{input, pipeline::RunCtx, window::WandaWindow};
use anyhow::Result;

/// 在窗口内指定坐标执行点击，支持逻辑像素或相对比例。
pub struct ClickWindowPos {
    pub pos: WindowPos,
}

#[derive(Clone, Copy)]
pub enum WindowPos {
    /// 直接使用窗口坐标系（左上角为 0,0）的偏移量。
    Logical { x: i32, y: i32 },
    /// 使用 0.0–1.0 的比例坐标，自动适配窗口大小。
    Ratio { x: f32, y: f32 },
}

impl ClickWindowPos {
    pub fn at_logical(x: i32, y: i32) -> Self {
        Self {
            pos: WindowPos::Logical { x, y },
        }
    }

    pub fn at_ratio(x: f32, y: f32) -> Self {
        Self {
            pos: WindowPos::Ratio { x, y },
        }
    }
}

impl Step for ClickWindowPos {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let (win_x, win_y) = window.position()?;
        let (win_w, win_h) = window.size()?;

        let (offset_x, offset_y) = match self.pos {
            WindowPos::Logical { x, y } => (x, y),
            WindowPos::Ratio { x, y } => {
                let clamped_x = x.clamp(0.0, 1.0);
                let clamped_y = y.clamp(0.0, 1.0);
                (
                    (win_w as f32 * clamped_x).round() as i32,
                    (win_h as f32 * clamped_y).round() as i32,
                )
            }
        };

        let screen_x = win_x + offset_x;
        let screen_y = win_y + offset_y;

        input::click_screen(screen_x, screen_y)?;
        ctx.invalidate();
        Ok(())
    }

    fn label(&self) -> &'static str {
        "ClickWindowPos"
    }
}
