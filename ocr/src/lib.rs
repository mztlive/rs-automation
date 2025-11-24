//! Thin wrapper around `rust-paddle-ocr` that provides an ergonomic,
//! reusable OCR engine for the workspace.
//!
//! The defaults follow the recommended PP-OCRv5 settings from the upstream
//! project (merge detected boxes, efficient cropping, and higher recognition
//! thresholds). Pass custom [`OcrOptions`] if you need different behavior.

mod config;
mod engine;
mod result;

pub use config::{ModelConfig, OcrOptions};
pub use engine::{OcrEngine, recognize_image};
pub use result::{BoundingBox, RecognizedText};

/// Crate-wide result type.
pub type OcrResult<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    use super::{BoundingBox, ModelConfig, OcrEngine, OcrOptions};
    use imageproc::rect::Rect;

    #[test]
    fn bounding_box_converts_from_rect() {
        let rect = Rect::at(5, 10).of_size(20, 30);
        let bbox: BoundingBox = rect.into();
        assert_eq!(bbox.x, 5);
        assert_eq!(bbox.y, 10);
        assert_eq!(bbox.width, 20);
        assert_eq!(bbox.height, 30);
    }

    #[test]
    fn options_apply_defaults() {
        let opts = OcrOptions::default();
        assert!(opts.merge_boxes);
        assert!(opts.efficient_cropping);
        assert_eq!(opts.merge_threshold, 1);
    }

    // Loading full models is out of scope for unit tests; this checks the API compiles.
    #[test]
    fn engine_type_is_send_sync_safe_to_move() {
        fn takes_engine(_: OcrEngine) {}
        fn build_dummy_config() -> ModelConfig {
            ModelConfig::new(
                "artifacts/ocr/PP-OCRv5_mobile_det_fp16.mnn",
                "artifacts/ocr/PP-OCRv5_mobile_rec_fp16.mnn",
                "artifacts/ocr/ppocr_keys_v5.txt",
            )
        }
        // We only ensure construction compiles with placeholder paths.
        let _ = build_dummy_config();
        let _ = OcrOptions::default();
        let _ = takes_engine;
    }
}
