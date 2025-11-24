use super::Step;
use crate::input;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;
use std::thread;
use std::time::Duration;

/// 通过鼠标滚轮横向滚动指定刻度（正值向右，负值向左）。
pub struct HorizontalScroll {
    pub lines: i32,
    pub settle_ms: u64,
}

impl Step for HorizontalScroll {
    fn run(&self, _window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        input::scroll_horizontal(self.lines)?;
        if self.settle_ms > 0 {
            thread::sleep(Duration::from_millis(self.settle_ms));
        }
        ctx.invalidate();
        Ok(())
    }

    fn label(&self) -> &'static str {
        "HorizontalScroll"
    }
}
