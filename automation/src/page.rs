use anyhow::{Result, anyhow};
use image_compat::RgbaImage as MlRgbaImage;
use ml::inference::{ModelConfig as MlModelConfig, PageClassifier as MlPageClassifier};
use serde::Deserialize;
use std::{cell::RefCell, fs, path::PathBuf};
use xcap::image::RgbaImage;

thread_local! {
    static PAGE_RECOGNIZER: RefCell<Option<PageRecognizer>> = RefCell::new(None);
}

/// 页面分类结果：要么未知，要么携带命中的页面名称。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Page {
    /// 未能根据模型输出匹配任何已知页面。
    Unknown,
    /// 成功匹配到的页面名（与 `assets/page.json` 的 name 对应）。
    Named(String),
}

/// `assets/page.json` 的整体配置结构。
#[derive(Debug, Clone, Deserialize)]
pub struct PageConfig {
    /// ML 模型加载信息。
    pub model: ModelSpec,
    /// 可选：页面锚点配置（继续沿用 OpenCV 模板用于精细操作）。
    #[serde(default)]
    pub pages: Vec<PageSpec>,
}

/// 模型配置：目录 + 权重/标签文件 + 预处理尺寸。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub dir: String,
    #[serde(default = "default_weights_file")]
    pub weights: String,
    #[serde(default = "default_labels_file")]
    pub labels: String,
    #[serde(default = "default_input_width")]
    pub input_width: u32,
    #[serde(default = "default_input_height")]
    pub input_height: u32,
    #[serde(default = "bool_true")]
    pub normalize: bool,
}

/// 单个页面的锚点配置。
#[derive(Debug, Clone, Deserialize)]
pub struct PageSpec {
    /// 模型输出的标签名/页面名。
    pub name: String,
    /// 此页面用于精细匹配的锚点集合。
    #[allow(dead_code)]
    #[serde(default)]
    pub anchors: Vec<AnchorSpec>,
}

/// 一个锚点（模板）及其阈值描述。
#[derive(Debug, Clone, Deserialize)]
pub struct AnchorSpec {
    /// 锚点名称，可在流水线上引用以执行模板匹配。
    pub name: String,
    /// 模板图片路径。
    pub image: String,
    /// 认为“命中”的最低匹配分数。
    pub threshold: f64,
    /// 当模板接近阈值时提供半分的容差。
    #[serde(default)]
    #[allow(dead_code)]
    pub near_margin: Option<f64>,
}

/// `classify` 的输出：包含页面与概率分数。
#[derive(Debug, Clone)]
pub struct PageScore {
    /// 识别出的页面。
    pub page: Page,
    /// 模型输出的置信度/概率。
    pub score: f64,
}

/// 负责管理配置与 Burn 模型实例。
pub struct PageRecognizer {
    cfg: PageConfig,
    classifier: MlPageClassifier,
}

impl PageRecognizer {
    fn from_config(cfg: PageConfig) -> Result<Self> {
        let ml_cfg = cfg.model.to_ml_config();
        let classifier = MlPageClassifier::load(&ml_cfg)?;
        Ok(Self { cfg, classifier })
    }

    /// 从指定路径加载配置与模型。
    pub fn from_path(path: &str) -> Result<Self> {
        let cfg = load_config(path)?;
        Self::from_config(cfg)
    }

    /// 对截图执行分类，并将概率映射为 `PageScore`。
    pub fn classify(&self, image: &RgbaImage) -> Result<PageScore> {
        let ml_image = convert_rgba(image);
        let prediction = self.classifier.predict_rgba(&ml_image)?;
        let known = self.find_page(&prediction.label).is_some();

        Ok(PageScore {
            page: if known {
                Page::Named(prediction.label.clone())
            } else {
                Page::Unknown
            },
            score: prediction.probability as f64,
        })
    }

    fn find_page(&self, page_name: &str) -> Option<&PageSpec> {
        self.cfg.pages.iter().find(|p| p.name == page_name)
    }

    fn find_anchor(&self, page_name: &str, anchor_name: &str) -> Option<&AnchorSpec> {
        self.find_page(page_name)
            .and_then(|page| page.anchors.iter().find(|a| a.name == anchor_name))
    }

    /// 查询某个页面的锚点配置，供模板匹配调用。
    #[allow(dead_code)]
    pub fn anchors_for(&self, page_name: &str) -> Option<&[AnchorSpec]> {
        self.find_page(page_name).map(|p| p.anchors.as_slice())
    }
}

impl ModelSpec {
    fn to_ml_config(&self) -> MlModelConfig {
        let dir = PathBuf::from(&self.dir);
        MlModelConfig {
            weight_path: dir.join(&self.weights),
            labels_path: dir.join(&self.labels),
            input_width: self.input_width,
            input_height: self.input_height,
            normalize: self.normalize,
        }
    }
}

fn default_weights_file() -> String {
    "page_net".to_string()
}

fn default_labels_file() -> String {
    "labels.json".to_string()
}

fn default_input_width() -> u32 {
    256
}

fn default_input_height() -> u32 {
    480
}

fn bool_true() -> bool {
    true
}

/// 初始化线程本地模型（幂等），供 `classify` 与 `WaitPage` 共享。
pub fn init(path: &str) -> Result<()> {
    PAGE_RECOGNIZER.with(|cell| {
        if cell.borrow().is_none() {
            let recognizer = PageRecognizer::from_path(path)?;
            *cell.borrow_mut() = Some(recognizer);
        }
        Ok(())
    })
}

fn with_recognizer<R>(f: impl FnOnce(&PageRecognizer) -> Result<R>) -> Result<R> {
    PAGE_RECOGNIZER.with(|cell| {
        let borrow = cell.borrow();
        let recognizer = borrow
            .as_ref()
            .ok_or_else(|| anyhow!("页面识别模型尚未初始化，请先调用 page::init"))?;
        f(recognizer)
    })
}

/// 对截图执行分类，使用已初始化的模型。
pub fn classify(image: &RgbaImage) -> Result<PageScore> {
    with_recognizer(|rec| rec.classify(image))
}

/// 从 JSON 文件读取页面配置。
pub fn load_config(path: &str) -> Result<PageConfig> {
    let text = fs::read_to_string(path)?;
    let cfg: PageConfig = serde_json::from_str(&text)?;
    Ok(cfg)
}

/// 根据页面名与锚点名获取模板配置。
pub fn anchor(page_name: &str, anchor_name: &str) -> Result<AnchorSpec> {
    with_recognizer(|rec| {
        rec.find_anchor(page_name, anchor_name)
            .cloned()
            .ok_or_else(|| anyhow!("页面 {page_name} 中不存在锚点 {anchor_name}"))
    })
}

fn convert_rgba(image: &RgbaImage) -> MlRgbaImage {
    let (w, h) = image.dimensions();
    MlRgbaImage::from_vec(w, h, image.as_raw().to_vec()).expect("截图尺寸与缓冲区长度匹配")
}
