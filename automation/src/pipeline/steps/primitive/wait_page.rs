use super::Step;
use crate::{page, pipeline::RunCtx, window::WandaWindow};
use anyhow::{Result, bail};
use std::{
    thread,
    time::{Duration, Instant},
};

/// 等待页面识别为目标页面（使用 Burn 训练的模型判断）。
///
/// 行为
/// - 周期性截图，使用 `page::classify` 识别当前页面；当命中目标页面名且得分 ≥ `min_score` 时返回成功。
/// - 超时未命中则报错。
/// - 为保证及时性，此步骤每次轮询都会主动截图，不使用缓存 `RunCtx`（避免点击后仍使用旧图）。
///
/// 参数
/// - `targets`: 允许的页面名称列表（命中任意一个即成功）。
/// - `min_score`: 认为“属于该页面”的最小分数（结合 page.json 的锚点设计）。
/// - `timeout_ms`/`poll_ms`: 超时与轮询间隔（毫秒）。
pub struct WaitPage {
    pub targets: &'static [&'static str],
    pub min_score: f64,
    pub timeout_ms: u64,
    pub poll_ms: u64,
}

impl Step for WaitPage {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let start = Instant::now();

        loop {
            let image = ctx.capture_rgba(window)?; // 始终抓取最新截图，等待阶段刷新缓存
            let res = page::classify(image)?;

            let matched = match &res.page {
                page::Page::Named(name) => {
                    self.targets.iter().any(|t| *t == name) && res.score >= self.min_score
                }
                page::Page::Unknown => false,
            };

            if matched {
                return Ok(());
            }

            if start.elapsed() >= Duration::from_millis(self.timeout_ms) {
                bail!(
                    "等待页面超时: 期待 {:?}, min_score={:.2}, 实际={:?} score={:.2}",
                    self.targets,
                    self.min_score,
                    res.page,
                    res.score,
                );
            }

            thread::sleep(Duration::from_millis(self.poll_ms));
        }
    }
}
