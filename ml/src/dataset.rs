use anyhow::{Context, Result, anyhow};
use burn::{
    data::dataloader::batcher::Batcher,
    data::dataset::Dataset,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use image::{DynamicImage, imageops::FilterType};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use std::{collections::HashMap, fs::File, path::PathBuf, sync::Arc};

const CHANNELS: usize = 3;

/// 读取 manifest 的基本配置。
#[derive(Clone, Debug)]
pub struct PageDatasetConfig {
    /// CSV 清单文件路径（包含 image,label 列）。
    pub manifest: PathBuf,
    /// 图像根目录，manifest 中的路径会在此目录下解析。
    pub image_root: PathBuf,
    /// 归一化后的目标宽度。
    pub width: u32,
    /// 归一化后的目标高度。
    pub height: u32,
    /// 是否将像素缩放到 `[0, 1]`。
    pub normalize: bool,
}

#[derive(Clone)]
struct DatasetEntry {
    relative_path: PathBuf,
    label_index: usize,
}

/// 单个样本，供 `Batcher` 组装成张量。
#[derive(Clone, Debug)]
pub struct PageSample {
    pub pixels: Vec<f32>,
    pub label: usize,
}

/// 批数据，包含图像与标签张量。
#[derive(Clone, Debug)]
pub struct PageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct PageDataset {
    root: Arc<PathBuf>,
    entries: Arc<Vec<DatasetEntry>>,
    labels: Arc<Vec<String>>,
    width: u32,
    height: u32,
    normalize: bool,
}

#[derive(Deserialize)]
struct ManifestRow {
    image: String,
    label: String,
}

impl PageDataset {
    /// 从 manifest 初始化数据集，同时构建标签映射。
    pub fn from_config(config: &PageDatasetConfig) -> Result<Self> {
        let file = File::open(&config.manifest)
            .with_context(|| format!("无法打开清单文件 {}", config.manifest.display()))?;
        let mut reader = csv::Reader::from_reader(file);

        let mut label_to_index = HashMap::new();
        let mut labels = Vec::new();
        let mut entries = Vec::new();

        for (row_idx, record) in reader.deserialize::<ManifestRow>().enumerate() {
            let record = record.with_context(|| format!("解析清单第 {} 行失败", row_idx + 1))?;
            let label_index = *label_to_index
                .entry(record.label.clone())
                .or_insert_with(|| {
                    labels.push(record.label.clone());
                    labels.len() - 1
                });

            let relative = PathBuf::from(record.image.trim());
            let full_path = config.image_root.join(&relative);
            if !full_path.exists() {
                return Err(anyhow!(
                    "清单中的图像不存在：{}（解析后路径：{}）",
                    relative.display(),
                    full_path.display()
                ));
            }

            entries.push(DatasetEntry {
                relative_path: relative,
                label_index,
            });
        }

        if entries.is_empty() {
            return Err(anyhow!("清单 {} 中没有可用样本", config.manifest.display()));
        }

        Ok(Self {
            root: Arc::new(config.image_root.clone()),
            entries: Arc::new(entries),
            labels: Arc::new(labels),
            width: config.width,
            height: config.height,
            normalize: config.normalize,
        })
    }

    fn with_entries(&self, entries: Vec<DatasetEntry>) -> Self {
        Self {
            root: self.root.clone(),
            entries: Arc::new(entries),
            labels: self.labels.clone(),
            width: self.width,
            height: self.height,
            normalize: self.normalize,
        }
    }

    /// 返回类别数量。
    pub fn num_classes(&self) -> usize {
        self.labels.len()
    }

    /// 标签名称有序列表。
    pub fn label_names(&self) -> Arc<Vec<String>> {
        self.labels.clone()
    }

    /// 返回 (宽, 高)。
    pub fn image_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// 将数据集随机划分为训练集与验证集。
    pub fn split(&self, val_fraction: f32, seed: u64) -> (Self, Self) {
        if val_fraction <= 0.0 {
            return (self.clone(), self.with_entries(Vec::new()));
        }

        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let val_count = ((indices.len() as f32) * val_fraction)
            .round()
            .clamp(0.0, indices.len() as f32) as usize;
        let val_count = val_count.min(indices.len());

        let (val_indices, train_indices) = indices.split_at(val_count);

        let to_entries = |slice: &[usize]| -> Vec<DatasetEntry> {
            slice.iter().map(|&idx| self.entries[idx].clone()).collect()
        };

        let train = self.with_entries(to_entries(train_indices));
        let val = self.with_entries(to_entries(val_indices));

        (train, val)
    }

    /// 创建一个批处理器，供 DataLoader 组装批次。
    pub fn batcher<B: Backend>(&self) -> PageBatcher<B> {
        PageBatcher::new(self.height as usize, self.width as usize)
    }
}

impl Dataset<PageSample> for PageDataset {
    fn get(&self, index: usize) -> Option<PageSample> {
        if index >= self.entries.len() {
            return None;
        }
        let entry = &self.entries[index];
        let full_path = self.root.join(&entry.relative_path);

        let image = image::open(&full_path)
            .unwrap_or_else(|err| panic!("读取图像 {} 失败：{err}", full_path.display()));

        let processed = preprocess_image(image, self.width, self.height, self.normalize);

        Some(PageSample {
            pixels: processed,
            label: entry.label_index,
        })
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn preprocess_image(image: DynamicImage, width: u32, height: u32, normalize: bool) -> Vec<f32> {
    let mut rgb = image.to_rgb8();
    if rgb.width() != width || rgb.height() != height {
        rgb = image::imageops::resize(&rgb, width, height, FilterType::Lanczos3);
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut data = vec![0.0f32; CHANNELS * width_usize * height_usize];

    for y in 0..height_usize {
        for x in 0..width_usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..CHANNELS {
                let value = pixel[c] as f32;
                let value = if normalize { value / 255.0 } else { value };
                let idx = c * width_usize * height_usize + y * width_usize + x;
                data[idx] = value;
            }
        }
    }

    data
}

/// 将单个样本堆叠成批处理张量。
#[derive(Clone)]
pub struct PageBatcher<B: Backend> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> PageBatcher<B> {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, PageSample, PageBatch<B>> for PageBatcher<B> {
    fn batch(&self, items: Vec<PageSample>, device: &B::Device) -> PageBatch<B> {
        let batch_size = items.len();
        let spatial = self.height * self.width;
        let mut pixels = Vec::with_capacity(batch_size * CHANNELS * spatial);
        let mut labels: Vec<i32> = Vec::with_capacity(batch_size);

        for sample in items {
            pixels.extend_from_slice(&sample.pixels);
            labels.push(sample.label as i32);
        }

        let images = Tensor::<B, 4>::from_data(
            TensorData::new(pixels, [batch_size, CHANNELS, self.height, self.width]),
            device,
        );

        let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [batch_size]), device);

        PageBatch { images, targets }
    }
}
