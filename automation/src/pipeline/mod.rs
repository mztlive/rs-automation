use crate::page;
use crate::window::{WandaWindow, WindowSelector};
use anyhow::{Result, anyhow};

mod context;
mod steps;
mod value;
mod wanda;
pub use context::*;
pub use steps::*;
pub use value::*;
#[allow(unused_imports)]
pub use wanda::*;

/// 将流水线步骤与窗口/页面配置打包，方便按名称选择执行。
pub struct PipelineBundle {
    pub name: String,
    pub page_config_path: String,
    pub window_selector: WindowSelector,
    pub pipeline: Pipeline,
}

impl PipelineBundle {
    pub fn new(
        name: impl Into<String>,
        page_config_path: impl Into<String>,
        window_selector: WindowSelector,
        pipeline: Pipeline,
    ) -> Self {
        Self {
            name: name.into(),
            page_config_path: page_config_path.into(),
            window_selector,
            pipeline,
        }
    }

    /// 初始化页面识别模型、查找并激活窗口，然后运行流水线步骤。
    pub fn run(&self, ctx: &mut RunCtx) -> Result<()> {
        println!("运行 pipeline [{}]", self.name);
        page::init(&self.page_config_path)?;
        let mut wanda_window = WandaWindow::new();
        wanda_window.find_window(&self.window_selector)?;
        wanda_window.activate()?;
        self.pipeline.run(&mut wanda_window, ctx)
    }
}

/// 根据名字选择流水线，便于扩展不同场景。
pub fn resolve_pipeline(name: &str, request: &BookingRequest) -> Result<PipelineBundle> {
    match name {
        "wanda" | "default" => Ok(wanda::wanda_pipeline(request)),
        other => Err(anyhow!("未知 pipeline：{other}")),
    }
}

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
