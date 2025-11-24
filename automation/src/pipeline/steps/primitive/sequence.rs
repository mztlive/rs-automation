use super::Step;
use crate::{pipeline::RunCtx, window::WandaWindow};
use anyhow::Result;

/// 复合步骤：将多个步骤按顺序组合成一个可复用的步骤。
///
/// 用途
/// - 将常用的动作序列封装成一个 `Sequence`，在不同页面或分支中复用，减少重复代码。
///
/// 行为
/// - 逐个执行内部的步骤，一旦某个步骤返回错误，将中断执行并向上传递错误。
pub struct Sequence {
    pub(crate) steps: Vec<Box<dyn Step + Send + Sync>>,
}

impl Sequence {
    /// 创建空序列。
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// 追加一个步骤并返回自身，便于链式构建。
    pub fn step(mut self, s: impl Step + Send + Sync + 'static) -> Self {
        self.steps.push(Box::new(s));
        self
    }

    /// 依次执行序列中的步骤。
    pub fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        for s in &self.steps {
            s.run(window, ctx)?;
        }
        Ok(())
    }
}

impl Step for Sequence {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        self.run(window, ctx)
    }
}
