use crate::pipeline::Step;

/// 简单的调试步骤：将消息打印到 stdout，不做任何自动化操作。
pub struct DebugStep {
    message: String,
}

impl DebugStep {
    /// 创建携带指定消息的调试步骤。
    pub fn new(message: &str) -> Self {
        DebugStep {
            message: message.to_string(),
        }
    }
}

impl Step for DebugStep {
    fn run(
        &self,
        _window: &mut crate::window::WandaWindow,
        _ctx: &mut crate::pipeline::RunCtx,
    ) -> anyhow::Result<()> {
        println!("{}", self.message);
        Ok(())
    }
}
