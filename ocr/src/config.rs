use std::path::{Path, PathBuf};

/// Required model paths for PaddleOCR.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    detection_model: PathBuf,
    recognition_model: PathBuf,
    keys_path: PathBuf,
}

impl ModelConfig {
    /// Create a new model configuration.
    pub fn new(
        detection_model: impl AsRef<Path>,
        recognition_model: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
    ) -> Self {
        Self {
            detection_model: detection_model.as_ref().to_path_buf(),
            recognition_model: recognition_model.as_ref().to_path_buf(),
            keys_path: keys_path.as_ref().to_path_buf(),
        }
    }

    /// Path to the detection model (`PP-OCRv5_mobile_det.mnn` or similar).
    pub fn detection_model(&self) -> &Path {
        &self.detection_model
    }

    /// Path to the recognition model (`PP-OCRv5_mobile_rec.mnn` or similar).
    pub fn recognition_model(&self) -> &Path {
        &self.recognition_model
    }

    /// Path to the keys/charset file (`ppocr_keys_v5.txt` or language specific).
    pub fn keys_path(&self) -> &Path {
        &self.keys_path
    }
}

/// Tunable parameters when running OCR.
#[derive(Debug, Clone, Copy)]
pub struct OcrOptions {
    /// Whether to merge nearby detected boxes (recommended for PP-OCRv5).
    pub merge_boxes: bool,
    /// Merge threshold passed to the detector.
    pub merge_threshold: i32,
    /// Use the faster cropping path from `rust-paddle-ocr`.
    pub efficient_cropping: bool,
    /// Minimum confidence for recognition.
    pub min_score: f32,
    /// Minimum confidence for punctuation recognition.
    pub punct_min_score: f32,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            merge_boxes: true,
            merge_threshold: 1,
            efficient_cropping: true,
            min_score: 0.6,
            punct_min_score: 0.1,
        }
    }
}
