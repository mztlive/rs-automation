use super::Step;
use crate::input;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;
use std::thread;
use std::time::Duration;

/// 简单的滚动步骤：通过鼠标滚轮上下滚动指定刻度。
pub struct Scroll {
    /// 正值向上，负值向下。刻度单位与系统/驱动相关，macOS 下建议使用较大的步长。
    pub lines: i32,
    /// 滚动后等待的毫秒数，给 UI 留出刷新时间。
    pub settle_ms: u64,
}

impl Step for Scroll {
    fn run(&self, _window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        input::scroll_vertical(self.lines)?;
        if self.settle_ms > 0 {
            thread::sleep(Duration::from_millis(self.settle_ms));
        }
        ctx.invalidate();
        Ok(())
    }

    fn label(&self) -> &'static str {
        "Scroll"
    }
}
