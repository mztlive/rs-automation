use super::Step;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::{Result, anyhow};
use std::thread;
use std::time::Duration;

/// 固定次数的循环原语：无条件执行指定步骤若干次。
pub struct LoopCount<S: Step + Send + Sync> {
    pub label: &'static str,
    pub times: usize,
    pub body: S,
    pub delay_ms: Option<u64>,
}

impl<S: Step + Send + Sync> Step for LoopCount<S> {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        if self.times == 0 {
            return Err(anyhow!("times must be > 0"));
        }

        for i in 1..=self.times {
            println!("[loop:{}] iter {}", self.label, i);
            self.body.run(window, ctx)?;
            ctx.invalidate();

            if i != self.times {
                if let Some(ms) = self.delay_ms {
                    thread::sleep(Duration::from_millis(ms));
                }
            }
        }

        Ok(())
    }

    fn label(&self) -> &'static str {
        self.label
    }
}
