use crate::model::PageNet;
use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::{Tensor, TensorData, activation::softmax, backend::Backend};
use image::{DynamicImage, RgbImage, RgbaImage, imageops::FilterType};
use std::{fs::File, path::PathBuf};

/// 默认使用 CPU（NdArray）后端，避免依赖特定图形 API，便于跨平台编译。
type InferenceBackend = burn::backend::ndarray::NdArray<f32>;

/// 模型加载配置：包含权重/标签路径与预处理超参数。
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub weight_path: PathBuf,
    pub labels_path: PathBuf,
    pub input_width: u32,
    pub input_height: u32,
    pub normalize: bool,
}

/// 模型输出：包含最可能的类别与完整概率分布。
#[derive(Debug, Clone)]
pub struct PagePrediction {
    pub label_index: usize,
    pub label: String,
    pub probability: f32,
    pub probabilities: Vec<(String, f32)>,
}

/// 封装 Burn 模型与预处理逻辑，可在运行时复用。
pub struct PageClassifier {
    device: <InferenceBackend as Backend>::Device,
    model: PageNet<InferenceBackend>,
    labels: Vec<String>,
    width: u32,
    height: u32,
    normalize: bool,
}

impl PageClassifier {
    /// 根据给定配置加载模型与标签映射。
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let device = <InferenceBackend as Backend>::Device::default();

        let labels_file = File::open(&config.labels_path)
            .with_context(|| format!("无法读取标签文件：{}", config.labels_path.display()))?;
        let labels: Vec<String> = serde_json::from_reader(labels_file)
            .with_context(|| format!("解析标签文件失败：{}", config.labels_path.display()))?;
        if labels.is_empty() {
            anyhow::bail!("标签列表为空：{}", config.labels_path.display());
        }

        let mut model = PageNet::<InferenceBackend>::new(
            &device,
            config.input_height as usize,
            config.input_width as usize,
            labels.len(),
        );
        model = model
            .load_file(
                &config.weight_path,
                &DefaultFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .with_context(|| format!("加载模型权重失败：{}", config.weight_path.display()))?;

        Ok(Self {
            device,
            model,
            labels,
            width: config.input_width,
            height: config.input_height,
            normalize: config.normalize,
        })
    }

    /// 对 RGB 图像执行推理。
    pub fn predict_rgb(&self, image: &RgbImage) -> Result<PagePrediction> {
        let tensor = self.preprocess(image);
        self.forward_tensor(tensor)
    }

    /// 对 RGBA 图像执行推理（自动转换为 RGB）。
    pub fn predict_rgba(&self, image: &RgbaImage) -> Result<PagePrediction> {
        let dynamic = DynamicImage::ImageRgba8(image.clone());
        let rgb = dynamic.to_rgb8();
        self.predict_rgb(&rgb)
    }

    /// 对任意动态图像执行推理。
    pub fn predict_dynamic(&self, image: &DynamicImage) -> Result<PagePrediction> {
        let rgb = image.to_rgb8();
        self.predict_rgb(&rgb)
    }

    fn preprocess(&self, image: &RgbImage) -> Tensor<InferenceBackend, 4> {
        let mut resized = image.clone();
        if resized.width() != self.width || resized.height() != self.height {
            resized =
                image::imageops::resize(&resized, self.width, self.height, FilterType::Lanczos3);
        }

        let width_usize = self.width as usize;
        let height_usize = self.height as usize;
        let mut data = vec![0.0f32; 3 * width_usize * height_usize];

        for y in 0..height_usize {
            for x in 0..width_usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let mut value = pixel[c] as f32;
                    if self.normalize {
                        value /= 255.0;
                    }
                    let idx = c * width_usize * height_usize + y * width_usize + x;
                    data[idx] = value;
                }
            }
        }

        Tensor::<InferenceBackend, 4>::from_data(
            TensorData::new(data, [1, 3, height_usize, width_usize]),
            &self.device,
        )
    }

    fn forward_tensor(&self, tensor: Tensor<InferenceBackend, 4>) -> Result<PagePrediction> {
        let logits = self.model.forward(tensor);
        let probs = softmax(logits.clone(), 1);
        let preds = logits.argmax(1);

        let class_index = preds
            .into_data()
            .into_vec::<i32>()
            .map_err(|err| anyhow::anyhow!("读取预测结果失败：{err:?}"))?[0]
            as usize;

        let probabilities = probs
            .into_data()
            .into_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("读取概率分布失败：{err:?}"))?;

        let label = self
            .labels
            .get(class_index)
            .cloned()
            .unwrap_or_else(|| format!("<unknown #{class_index}>"));
        let probability = probabilities.get(class_index).copied().unwrap_or(0.0);

        let distribution = self
            .labels
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let prob = probabilities.get(idx).copied().unwrap_or(0.0);
                (name.clone(), prob)
            })
            .collect();

        Ok(PagePrediction {
            label_index: class_index,
            label,
            probability,
            probabilities: distribution,
        })
    }
}
