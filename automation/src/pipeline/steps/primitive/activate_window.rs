use super::Step;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;

/// 激活/置前目标窗口的步骤。
///
/// 功能
/// - 将与自动化目标相关的窗口置于最前方，以保证后续的截图、找图与点击不会误作用到其他窗口。
///
/// 实现细节
/// - 内部调用 `WandaWindow::activate()`，在 macOS 上通过 AppleScript 先按 PID 置前，失败则按应用名激活。
/// - 调用后会短暂等待（在 `activate()` 内部实现），以确保窗口已经到前台。
///
/// 失败场景
/// - 若未授予“辅助功能”权限，可能无法成功置前；该步骤会返回错误，由上层处理。
pub struct ActivateWindow;

impl Step for ActivateWindow {
    fn run(&self, window: &mut WandaWindow, _ctx: &mut RunCtx) -> Result<()> {
        window.activate()
    }
}
