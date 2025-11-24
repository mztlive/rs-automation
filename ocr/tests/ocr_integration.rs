use ocr::{ModelConfig, OcrEngine};

// Heavy test that loads real OCR models; run with:
// cargo test -p ocr -- --ignored
#[test]
#[ignore = "loads real PP-OCRv5 models; enable when models are available locally"]
fn runs_ocr_pipeline_against_blank_image() {
    let config = ModelConfig::new(
        "/Users/huangjiajiang/Development/wanda-automation/artifacts/ocr/PP-OCRv5_mobile_det_fp16.mnn",
        "/Users/huangjiajiang/Development/wanda-automation/artifacts/ocr/PP-OCRv5_mobile_rec_fp16.mnn",
        "/Users/huangjiajiang/Development/wanda-automation/artifacts/ocr/ppocr_keys_v5.txt",
    );

    let mut engine = OcrEngine::new(config).expect("engine builds with fp16 models");

    // blank image should yield no detections
    let img = image::open(
        "/Users/huangjiajiang/Development/wanda-automation/assets/raw/movie-list/list1.png",
    )
    .unwrap();
    let results = engine
        .recognize_image(&img)
        .expect("ocr pipeline should run without error");

    assert!(
        !results.is_empty(),
        "expected no detections on a blank image"
    );
}
