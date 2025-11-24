use super::Step;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;
use std::{thread, time::Duration};

/// 固定时长的等待/休眠步骤。
///
/// 用途
/// - 在页面切换、弹窗出现、动画过渡等场景，插入一个短暂的时间缓冲，避免立刻找图导致误判。
///
/// 参数
/// - `SleepMs(ms)`: 休眠的毫秒数。
///
/// 注意
/// - 此步骤为“硬等待”，不做条件判断。若需要“等待某条件满足”，请使用 `WaitTemplate`。
pub struct SleepMs(pub u64);

impl Step for SleepMs {
    fn run(&self, _window: &mut WandaWindow, _ctx: &mut RunCtx) -> Result<()> {
        thread::sleep(Duration::from_millis(self.0));
        Ok(())
    }
}
