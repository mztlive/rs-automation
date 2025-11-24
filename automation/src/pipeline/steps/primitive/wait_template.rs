use super::Step;
use crate::pipeline::RunCtx;
use crate::{actions, window::WandaWindow};
use anyhow::{Result, bail};
use std::{
    thread,
    time::{Duration, Instant},
};

/// 等待模板出现或消失的步骤（支持超时与轮询）。
///
/// 功能
/// - 周期性截图并进行模板匹配，根据 `appear` 决定等待条件：
///   - `appear=true`：等待分数 ≥ 阈值（出现/稳定可见）
///   - `appear=false`：等待分数 < 阈值（消失/不可见）
/// - 若在 `timeout_ms` 内未满足条件，返回超时错误。
///
/// 参数
/// - `template`：模板图片路径。
/// - `threshold`：匹配阈值，建议 0.85–0.95。
/// - `appear`：等待出现或消失。
/// - `timeout_ms`：超时时间毫秒。
/// - `poll_ms`：轮询间隔毫秒。
#[allow(dead_code)]
pub struct WaitTemplate {
    pub template: &'static str,
    pub threshold: f64,
    pub appear: bool, // true: wait until present; false: wait until gone
    pub timeout_ms: u64,
    pub poll_ms: u64,
}

impl Step for WaitTemplate {
    fn run(&self, window: &mut WandaWindow, _ctx: &mut RunCtx) -> Result<()> {
        let start = Instant::now();
        loop {
            let image = window.capture()?;
            let (w, h) = window.size()?;
            let (x, y) = window.position()?;
            let ((_sx, _sy), score) =
                actions::find_template_screen_center(&image, self.template, (w, h), (x, y))?;

            let met = if self.appear {
                score >= self.threshold
            } else {
                score < self.threshold
            };
            if met {
                return Ok(());
            }

            if start.elapsed() >= Duration::from_millis(self.timeout_ms) {
                bail!(
                    "等待超时: template={}, expect {} threshold {:.3}",
                    self.template,
                    if self.appear {
                        "appear >="
                    } else {
                        "disappear <"
                    },
                    self.threshold
                );
            }
            thread::sleep(Duration::from_millis(self.poll_ms));
        }
    }
}
