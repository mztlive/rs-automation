use anyhow::anyhow;

use crate::pipeline::{Scroll, Step, steps::primitive};

pub struct ScrollThenOcrLoop {
    pub patterns: String,
    pub max_attempts: u32,
}

impl Step for ScrollThenOcrLoop {
    fn run(
        &self,
        window: &mut crate::window::WandaWindow,
        ctx: &mut crate::pipeline::RunCtx,
    ) -> anyhow::Result<()> {
        if self.max_attempts == 0 {
            return Err(anyhow!("max_attempts must be > 0"));
        }

        let sequence = primitive::Sequence::new()
            .step(primitive::LoopCount {
                label: "scroll top top",
                times: 10,
                body: primitive::Scroll {
                    lines: -10,
                    settle_ms: 150,
                },
                delay_ms: Some(150),
            })
            .step(primitive::LoopUntil {
                label: "find-text",
                cond: primitive::Condition::TextContains {
                    pattern: self.patterns.clone(),
                    case_sensitive: false,
                },
                on_miss: Scroll {
                    lines: 10,
                    settle_ms: 150,
                },
                max_iters: self.max_attempts as usize,
                delay_ms: Some(100),
            });

        sequence.run(window, ctx)
    }
}
