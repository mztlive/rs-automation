use super::Step;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::{Result, anyhow};
use std::thread;
use std::time::Duration;

/// 循环执行：如果条件未满足则跑一段“补救”序列（如滚动），直到命中或超出尝试次数。
pub struct LoopUntil<S: Step + Send + Sync> {
    pub label: &'static str,
    pub cond: super::condition::Condition,
    pub on_miss: S,
    pub max_iters: usize,
    pub delay_ms: Option<u64>,
}

impl<S: Step + Send + Sync> Step for LoopUntil<S> {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        if self.max_iters == 0 {
            return Err(anyhow!("max_iters must be > 0"));
        }

        for i in 1..=self.max_iters {
            let report = self.cond.eval(window, ctx)?;
            if let Some(info) = report.info.as_deref() {
                println!("[loop:{}] iter {} cond => {}", self.label, i, info);
            }

            if report.matched {
                println!("[loop:{}] matched at iter {}", self.label, i);
                return Ok(());
            }

            if i == self.max_iters {
                break;
            }

            println!("[loop:{}] miss at iter {}, running on_miss", self.label, i);
            self.on_miss.run(window, ctx)?;
            ctx.invalidate();

            if let Some(ms) = self.delay_ms {
                thread::sleep(Duration::from_millis(ms));
            }
        }

        Err(anyhow!(
            "[loop:{}] condition not met after {} iterations",
            self.label,
            self.max_iters
        ))
    }

    fn label(&self) -> &'static str {
        self.label
    }
}
