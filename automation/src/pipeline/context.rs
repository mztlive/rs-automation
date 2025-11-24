use crate::vision;
use crate::window::WandaWindow;
use anyhow::Result;
use ocr::{OcrEngine, RecognizedText};
use opencv::core::Mat;
use xcap::image::RgbaImage;

/// OCR 结果及其原始截图尺寸（用于窗口->屏幕坐标换算）。
pub struct OcrSnapshot {
    pub results: Vec<RecognizedText>,
    pub width: u32,
    pub height: u32,
}

/// 用户在 CLI 输入的订票信息。
#[derive(Debug, Clone)]
pub struct BookingRequest {
    pub movie_name: String,
    pub show_date: String,
    pub cinema_name: String,
    pub show_time: String,
}

impl BookingRequest {
    pub fn new(
        movie_name: impl Into<String>,
        show_date: impl Into<String>,
        cinema_name: impl Into<String>,
        show_time: impl Into<String>,
    ) -> Self {
        Self {
            movie_name: movie_name.into(),
            show_date: show_date.into(),
            cinema_name: cinema_name.into(),
            show_time: show_time.into(),
        }
    }
}

/// 运行上下文：在流水线执行过程中缓存截图，供各步骤复用。
/// 步骤可选择：
/// - 使用 `ensure_rgba/ensure_bgr` 复用缓存（减少频繁截图与转换）
/// - 主动调用 `invalidate` 使缓存失效（例如点击后 UI 变化）
/// - 直接绕过上下文自行截图（满足“步骤可自行决定是否使用缓存”的设计）
pub struct RunCtx {
    rgba: Option<RgbaImage>,
    bgr: Option<Mat>,
    ocr: Option<OcrSnapshot>,
    ocr_engine: Option<OcrEngine>,
    decisions: Vec<DecisionRecord>,
    booking: Option<BookingRequest>,
}

impl Default for RunCtx {
    fn default() -> Self {
        Self {
            rgba: None,
            bgr: None,
            ocr: None,
            ocr_engine: None,
            decisions: Vec::new(),
            booking: None,
        }
    }
}

impl RunCtx {
    /// 创建包含订票信息的上下文。
    pub fn with_booking_request(request: BookingRequest) -> Self {
        let mut ctx = Self::default();
        ctx.booking = Some(request);
        ctx
    }

    /// 主动使缓存失效，以便下一次调用重新截图/转换。
    pub fn invalidate(&mut self) {
        self.rgba = None;
        self.bgr = None;
        self.ocr = None;
    }

    /// 确保缓存中包含最新一次窗口截图的 RGBA 版本，必要时触发 `capture`。
    pub fn ensure_rgba(&mut self, window: &mut WandaWindow) -> Result<&RgbaImage> {
        if self.rgba.is_none() {
            self.rgba = Some(window.capture()?);
        }

        Ok(self.rgba.as_ref().unwrap())
    }

    /// 主动截图并刷新缓存，确保返回值总是最新画面。
    pub fn capture_rgba(&mut self, window: &mut WandaWindow) -> Result<&RgbaImage> {
        let img = window.capture()?;
        self.set_rgba(img);
        Ok(self.rgba.as_ref().unwrap())
    }

    #[allow(dead_code)]
    /// 确保缓存中包含最新截图的 BGR Mat 版本，必要时基于 RGBA 缓存转换。
    pub fn ensure_bgr(&mut self, window: &mut WandaWindow) -> Result<&Mat> {
        if self.bgr.is_none() {
            let rgba = self.ensure_rgba(window)?;
            let bgr = vision::rgba_to_bgr(rgba)?;
            self.bgr = Some(bgr);
        }

        Ok(self.bgr.as_ref().unwrap())
    }

    /// 直接注入一张 RGBA 截图，并使 BGR 缓存过期。
    pub fn set_rgba(&mut self, img: RgbaImage) {
        self.rgba = Some(img);
        self.bgr = None;
        self.ocr = None;
    }

    /// 尝试从缓存获取 OCR 结果。
    pub fn ocr_results(&self) -> Option<&[RecognizedText]> {
        self.ocr.as_ref().map(|s| s.results.as_slice())
    }

    /// 返回 OCR 结果对应的截图尺寸（像素）。
    pub fn ocr_dims(&self) -> Option<(u32, u32)> {
        self.ocr.as_ref().map(|s| (s.width, s.height))
    }

    /// 写入 OCR 结果缓存。
    pub fn set_ocr_results(&mut self, results: Vec<RecognizedText>, dims: (u32, u32)) {
        self.ocr = Some(OcrSnapshot {
            results,
            width: dims.0,
            height: dims.1,
        });
    }

    /// 获取或构建 OCR 引擎。调用方提供构建闭包，以便控制模型路径等。
    pub fn ensure_ocr_engine<F>(&mut self, build: F) -> Result<&mut OcrEngine>
    where
        F: FnOnce() -> Result<OcrEngine>,
    {
        if self.ocr_engine.is_none() {
            self.ocr_engine = Some(build()?);
        }
        Ok(self.ocr_engine.as_mut().expect("OCR engine must exist"))
    }

    /// 记录条件分支的命中情况，供外部观测。
    pub fn record_decision(&mut self, label: &str, branch: DecisionBranch) {
        self.decisions.push(DecisionRecord {
            label: label.to_string(),
            branch,
        });
    }

    /// 返回已记录的条件分支命中列表。
    #[allow(dead_code)]
    pub fn decisions(&self) -> &[DecisionRecord] {
        &self.decisions
    }

    /// 写入 CLI 输入的订票信息。
    pub fn set_booking_request(&mut self, request: BookingRequest) {
        self.booking = Some(request);
    }

    /// 读取订票信息（如果存在）。
    pub fn booking_request(&self) -> Option<&BookingRequest> {
        self.booking.as_ref()
    }
}

/// 条件分支命中记录。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    pub label: String,
    pub branch: DecisionBranch,
}

/// 条件分支命中类型。
#[derive(Debug, Clone, Copy)]
pub enum DecisionBranch {
    Then,
    Else,
    Skipped,
}
