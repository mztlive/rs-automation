use super::Step;
use crate::{pipeline::RunCtx, window::WandaWindow};
use anyhow::{Result, anyhow};
use std::{thread, time::Duration};

/// 为任意步骤提供可配置的重试逻辑。
pub struct RetryStep<S>
where
    S: Step + Send + Sync + 'static,
{
    label: &'static str,
    inner: S,
    attempts: usize,
    delay: Duration,
}

impl<S> RetryStep<S>
where
    S: Step + Send + Sync + 'static,
{
    pub fn new(label: &'static str, inner: S, attempts: usize, delay: Duration) -> Self {
        assert!(attempts > 0, "重试次数需大于 0");
        Self {
            label,
            inner,
            attempts,
            delay,
        }
    }
}

impl<S> Step for RetryStep<S>
where
    S: Step + Send + Sync + 'static,
{
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let mut last_err = None;
        for attempt in 1..=self.attempts {
            match self.inner.run(window, ctx) {
                Ok(_) => return Ok(()),
                Err(err) if attempt < self.attempts => {
                    println!(
                        "[retry:{}] attempt {}/{} failed: {:?}",
                        self.label, attempt, self.attempts, err
                    );
                    last_err = Some(err);
                    thread::sleep(self.delay);
                }
                Err(err) => return Err(err),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("{} 执行失败", self.label)))
    }

    fn label(&self) -> &'static str {
        self.label
    }
}
