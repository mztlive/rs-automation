use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;
use std::any::type_name;

/// 所有步骤类型的统一接口。
///
/// 每个步骤实现 `run`，在给定窗口上下文中执行一次原子操作（如：置前、等待、点击）。
/// - 步骤应当是“可重入”的：多次调用不会产生未定义副作用。
/// - 步骤应当处理自身需要的重试/等待逻辑，并在失败时返回错误。
/// - 步骤不应捕获致命错误（例如权限缺失），应向上传递，交由调用方处理。
pub trait Step {
    /// 执行步骤。
    /// - `window`：窗口上下文
    /// - `ctx`：运行上下文，包含可复用的截图缓存；步骤可选择使用或忽略
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()>;

    /// 返回步骤名称，默认使用类型名称，可在实现中重写以输出更友好的 label。
    fn label(&self) -> &'static str {
        type_name::<Self>()
    }
}

pub mod activate_window;
pub mod click_anchor;
pub mod click_ocr_match;
pub mod click_window_pos;
pub mod condition;
pub mod debug;
pub mod horizontal_scroll;
pub mod loop_count;
pub mod loop_until;
pub mod move_mouse;
pub mod retry;
pub mod scroll;
pub mod sequence;
pub mod sleep_ms;
pub mod text_match;
pub mod wait_page;
pub mod wait_template;

pub use activate_window::ActivateWindow;
pub use click_anchor::{AnchorClickPos, ClickAnchor};
pub use click_ocr_match::ClickOcrMatch;
pub use click_window_pos::{ClickWindowPos, WindowPos};
#[allow(unused_imports)]
pub use condition::{Condition, ConditionalStep};
pub use debug::DebugStep;
pub use horizontal_scroll::HorizontalScroll;
#[allow(unused_imports)]
pub use loop_count::LoopCount;
pub use loop_until::LoopUntil;
pub use move_mouse::{GridPos, MoveMouse};
pub use retry::RetryStep;
pub use scroll::Scroll;
pub use sequence::Sequence;
pub use sleep_ms::SleepMs;
pub use wait_page::WaitPage;
#[allow(unused_imports)]
pub use wait_template::WaitTemplate;
