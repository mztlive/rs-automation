use crate::OcrResult;
use crate::config::{ModelConfig, OcrOptions};
use crate::result::{BoundingBox, RecognizedText};
use anyhow::Context;
use image::DynamicImage;
use imageproc::rect::Rect;
use rust_paddle_ocr::efficient_cropping::{EfficientCropper, ImageRef};
use rust_paddle_ocr::{Det, Rec};
use std::path::Path;

/// High-level OCR engine wrapping `rust-paddle-ocr`.
///
/// Holds the detection and recognition models so they can be reused across
/// multiple calls without paying the load cost each time.
pub struct OcrEngine {
    det: Det,
    rec: Rec,
    options: OcrOptions,
}

impl OcrEngine {
    /// Build an engine with default options.
    pub fn new(config: ModelConfig) -> OcrResult<Self> {
        Self::with_options(config, OcrOptions::default())
    }

    /// Build an engine with custom options.
    pub fn with_options(config: ModelConfig, options: OcrOptions) -> OcrResult<Self> {
        let det = Det::from_file(config.detection_model())
            .context("failed to load detection model")?
            .with_merge_boxes(options.merge_boxes)
            .with_merge_threshold(options.merge_threshold);

        let rec = Rec::from_file(config.recognition_model(), config.keys_path())
            .context("failed to load recognition model")?
            .with_min_score(options.min_score)
            .with_punct_min_score(options.punct_min_score);

        Ok(Self { det, rec, options })
    }

    /// Run OCR on an image file.
    pub fn recognize_path(
        &mut self,
        image_path: impl AsRef<Path>,
    ) -> OcrResult<Vec<RecognizedText>> {
        let image = image::open(&image_path)
            .with_context(|| format!("failed to open image at {:?}", image_path.as_ref()))?;
        self.recognize_image(&image)
    }

    /// Run OCR on an already loaded image.
    pub fn recognize_image(&mut self, image: &DynamicImage) -> OcrResult<Vec<RecognizedText>> {
        let rects = self
            .det
            .find_text_rect(image)
            .context("text detection failed")?;

        if rects.is_empty() {
            return Ok(Vec::new());
        }

        let crops = self.crop_regions(image, &rects);

        let mut results = Vec::with_capacity(rects.len());
        for (rect, crop) in rects.into_iter().zip(crops.into_iter()) {
            let bbox = BoundingBox::from(rect);
            let text = self
                .rec
                .predict_str(&crop)
                .context("text recognition failed")?;
            results.push(RecognizedText::new(bbox, text));
        }

        Ok(results)
    }

    fn crop_regions(&self, image: &DynamicImage, rects: &[Rect]) -> Vec<DynamicImage> {
        if !self.options.efficient_cropping {
            return rects
                .iter()
                .map(|rect| {
                    image.crop_imm(
                        rect.left() as u32,
                        rect.top() as u32,
                        rect.width(),
                        rect.height(),
                    )
                })
                .collect();
        }

        // Use the upstream optimized cropper to reduce clones for multiple boxes.
        let image_ref = ImageRef::from(image.clone());
        match rects.len() {
            1 => vec![EfficientCropper::smart_crop(&image_ref, &rects[0])],
            2..=8 => EfficientCropper::parallel_batch_crop(&image_ref, rects),
            _ => EfficientCropper::optimized_batch_crop(&image_ref, rects),
        }
    }
}

impl From<Rect> for BoundingBox {
    fn from(rect: Rect) -> Self {
        let x = rect.left().max(0) as u32;
        let y = rect.top().max(0) as u32;
        Self {
            x,
            y,
            width: rect.width(),
            height: rect.height(),
        }
    }
}

/// Convenience function to run OCR without manually managing the engine.
pub fn recognize_image(
    config: ModelConfig,
    image_path: impl AsRef<Path>,
    options: Option<OcrOptions>,
) -> OcrResult<Vec<RecognizedText>> {
    let mut engine = match options {
        Some(opts) => OcrEngine::with_options(config, opts)?,
        None => OcrEngine::new(config)?,
    };
    engine.recognize_path(image_path)
}
