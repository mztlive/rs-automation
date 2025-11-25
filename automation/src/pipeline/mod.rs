use crate::window::WandaWindow;
use anyhow::{Result, anyhow};

mod context;
mod default;
mod steps;
mod value;
pub use context::*;
pub use default::*;
pub use steps::*;
pub use value::*;

/// 流水线：由若干 `Step` 组成，按顺序执行，用于描述某页面下的一次完整自动化流程。
pub struct Pipeline {
    steps: Vec<Box<dyn Step + Send + Sync>>, // Send+Sync to ease future threading
}

impl Pipeline {
    /// 创建空流水线。
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// 追加一个步骤，返回自身以便链式调用。
    pub fn step(mut self, s: impl Step + Send + Sync + 'static) -> Self {
        self.steps.push(Box::new(s));
        self
    }

    /// 依次执行流水线中的所有步骤，任一步骤错误将被向上传递。
    /// 会立刻执行一次截图，确保有内容可以往下传递
    pub fn run(&self, wanda_window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let mut i = 0;
        for step in &self.steps {
            println!("[step {:02}] {}", i, step.label());
            step.run(wanda_window, ctx)
                .map_err(|err| anyhow!("step {} throw error: {:?}", i, err))?;

            i += 1;
        }
        if !ctx.decisions().is_empty() {
            println!("决策路径:");
            for record in ctx.decisions() {
                println!("  - {} => {:?}", record.label, record.branch);
            }
        }
        Ok(())
    }
}

/// 判断页面名是否属于一级页面（存在底部导航）。
fn is_first_level_page(name: &str) -> bool {
    matches!(
        name,
        "Home"
            | "home"
            | "\u{9996}\u{9875}"
            | "user"
            | "User"
            | "mall"
            | "Mall"
            | "movie-list"
            | "movie"
    )
}
