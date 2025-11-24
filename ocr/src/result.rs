/// Bounding box of a detected text region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BoundingBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// OCR output for a single detected region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecognizedText {
    pub bbox: BoundingBox,
    pub text: String,
}

impl RecognizedText {
    pub fn new(bbox: BoundingBox, text: String) -> Self {
        Self { bbox, text }
    }
}
