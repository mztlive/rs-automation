use super::Step;
use crate::pipeline::{DecisionBranch, RunCtx};
use crate::{actions, page, window::WandaWindow};
use anyhow::{Result, anyhow};
use image::{DynamicImage, GenericImageView, RgbaImage};
use ocr::{ModelConfig, OcrEngine};
use std::fmt::Write;

/// 条件枚举：在运行时对当前界面进行判断，以便在流水线中进行分支控制。
///
/// 支持：
/// - `Always`：恒为真，用于调试或占位。
/// - `TemplateAbove { template, threshold }`：模板匹配分数 ≥ 阈值。
/// - `TemplateBelow { template, threshold }`：模板匹配分数 < 阈值。
/// - `AnchorAbove { page, anchor }`：根据 page.json 中的锚点模板判断分数 ≥ 阈值。
/// - `AnchorBelow { page, anchor }`：根据 page.json 中的锚点模板判断分数 < 阈值。
/// - `TextContains { pattern, case_sensitive }`：OCR 识别结果与给定文本相似。

pub enum Condition {
    #[allow(dead_code)]
    Always,

    AnchorAbove {
        page: &'static str,
        anchor: &'static str,
    },
    #[allow(dead_code)]
    AnchorBelow {
        page: &'static str,
        anchor: &'static str,
    },

    #[allow(dead_code)]
    /// 在当前界面做一次 OCR，匹配目标文本。
    TextContains {
        /// 期望匹配的字符串。
        pattern: String,
        /// 是否区分大小写（默认建议 false）。
        case_sensitive: bool,
    },
}

impl Condition {
    /// 在当前窗口上评估条件，返回是否满足。
    pub fn eval(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<ConditionReport> {
        match self {
            Condition::Always => Ok(ConditionReport::matched("always true")),

            Condition::AnchorAbove { page, anchor } => {
                let image = ctx.capture_rgba(window)?;
                let spec = page::anchor(page, anchor)?;
                let (w, h) = window.size()?;
                let (x, y) = window.position()?;
                let ((_sx, _sy), score) = actions::find_template_screen_center(
                    &image,
                    spec.image.as_str(),
                    (w, h),
                    (x, y),
                )?;
                let matched = score >= spec.threshold;
                let info = format!(
                    "anchor {}::{} score {:.4} threshold {:.4}",
                    page, anchor, score, spec.threshold
                );
                Ok(ConditionReport::new(matched, info))
            }
            Condition::AnchorBelow { page, anchor } => {
                let image = ctx.capture_rgba(window)?;
                let spec = page::anchor(page, anchor)?;
                let (w, h) = window.size()?;
                let (x, y) = window.position()?;
                let ((_sx, _sy), score) = actions::find_template_screen_center(
                    &image,
                    spec.image.as_str(),
                    (w, h),
                    (x, y),
                )?;
                let matched = score < spec.threshold;
                let info = format!(
                    "anchor {}::{} score {:.4} threshold {:.4} (below)",
                    page, anchor, score, spec.threshold
                );
                Ok(ConditionReport::new(matched, info))
            }
            Condition::TextContains {
                pattern,
                case_sensitive,
            } => {
                // 尝试复用上一帧的 OCR 结果。
                let recognized = if let Some(cached) = ctx.ocr_results() {
                    cached.to_vec()
                } else {
                    let rgba = ctx.capture_rgba(window)?;
                    let dyn_img = to_dynamic_image(rgba)?;
                    let engine = ctx.ensure_ocr_engine(build_ocr_engine)?;
                    let results = engine.recognize_image(&dyn_img)?;
                    let dims = dyn_img.dimensions();
                    ctx.set_ocr_results(results.clone(), dims);
                    results
                };

                println!(
                    "recognized: {:?}",
                    recognized
                        .iter()
                        .map(|i| i.text.clone())
                        .collect::<Vec<_>>()
                );

                let matched = has_match(&recognized, pattern, *case_sensitive);
                let mut info = String::new();
                let _ = write!(
                    &mut info,
                    "ocr boxes={} matched={} pattern={:?}",
                    recognized.len(),
                    matched,
                    pattern
                );
                Ok(ConditionReport::new(matched, info))
            }
        }
    }
}

/// 条件评估结果及调试信息。
pub struct ConditionReport {
    pub matched: bool,
    pub info: Option<String>,
}

impl ConditionReport {
    fn new(matched: bool, info: impl Into<String>) -> Self {
        Self {
            matched,
            info: Some(info.into()),
        }
    }

    fn matched(info: impl Into<String>) -> Self {
        Self {
            matched: true,
            info: Some(info.into()),
        }
    }
}

/// 条件分支步骤：根据 `cond` 的真假，分别执行 `then_seq` 或 `else_seq`。
/// - `then_seq`：条件为真时执行的步骤序列。
/// - `else_seq`：条件为假时执行的步骤序列（可选）。
pub struct ConditionalStep<S: Step + Send + Sync> {
    /// 便于观测/记录的标签，建议使用具描述性的名称。
    pub label: &'static str,
    pub cond: Condition,
    pub then_seq: S,
    pub else_seq: Option<S>,
}

impl<S: Step + Send + Sync> Step for ConditionalStep<S> {
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        let report = self.cond.eval(window, ctx)?;
        if let Some(info) = report.info.as_deref() {
            println!("[cond:{}] {}", self.label, info);
        }

        if report.matched {
            ctx.record_decision(self.label, DecisionBranch::Then);
            println!("[cond:{}] then branch", self.label);
            self.then_seq.run(window, ctx)
        } else if let Some(es) = &self.else_seq {
            ctx.record_decision(self.label, DecisionBranch::Else);
            println!("[cond:{}] else branch", self.label);
            es.run(window, ctx)
        } else {
            ctx.record_decision(self.label, DecisionBranch::Skipped);
            println!("[cond:{}] skipped", self.label);
            Ok(())
        }
    }
}

pub const DET_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../artifacts/ocr/PP-OCRv5_mobile_det_fp16.mnn"
);
pub const REC_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../artifacts/ocr/PP-OCRv5_mobile_rec_fp16.mnn"
);
pub const KEYS_TXT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../artifacts/ocr/ppocr_keys_v5.txt"
);

pub fn build_ocr_engine() -> Result<OcrEngine> {
    let models = ModelConfig::new(DET_MODEL, REC_MODEL, KEYS_TXT);
    OcrEngine::new(models)
}

fn has_match(results: &[ocr::RecognizedText], pattern: &str, case_sensitive: bool) -> bool {
    results
        .iter()
        .any(|r| super::text_match::has_similar(&r.text, pattern, case_sensitive))
}

pub fn to_dynamic_image(img: &xcap::image::RgbaImage) -> Result<DynamicImage> {
    let (w, h) = img.dimensions();
    let flat = img.as_flat_samples();
    let buf = RgbaImage::from_raw(w, h, flat.as_slice().to_vec())
        .ok_or_else(|| anyhow!("failed to convert screenshot to DynamicImage"))?;
    Ok(DynamicImage::ImageRgba8(buf))
}
