use crate::{
    dataset::{PageBatch, PageDataset, PageDatasetConfig, PageSample},
    model::PageNet,
};
use anyhow::{Context, Result, anyhow};
use burn::{
    backend::Autodiff,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    data::dataset::Dataset as _,
    module::{AutodiffModule, Module},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::Backend,
};
use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(target_os = "windows")]
use burn::backend::ndarray::NdArray;
#[cfg(not(target_os = "windows"))]
use burn::backend::wgpu::{self, graphics::AutoGraphicsApi, RuntimeOptions, Wgpu};

#[cfg(not(target_os = "windows"))]
type InnerBackend = Wgpu<f32>;
#[cfg(target_os = "windows")]
type InnerBackend = NdArray<f32>;
type TrainBackend = Autodiff<InnerBackend>;

#[cfg(not(target_os = "windows"))]
fn init_backend(device: &<TrainBackend as Backend>::Device) {
    let _ = wgpu::init_setup::<AutoGraphicsApi>(device, RuntimeOptions::default());
}

#[cfg(target_os = "windows")]
fn init_backend(_: &<TrainBackend as Backend>::Device) {}

/// 训练时的超参数。
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// 数据加载配置。
    pub dataset: PageDatasetConfig,
    /// 输出目录（模型与标签映射会写在这里）。
    pub output_dir: PathBuf,
    /// batch 大小。
    pub batch_size: usize,
    /// epoch 数。
    pub epochs: usize,
    /// 学习率。
    pub learning_rate: f64,
    /// 验证集比例（0-1）。
    pub val_split: f32,
    /// 随机种子。
    pub seed: u64,
}

/// 训练产物：模型与标签映射路径。
#[derive(Clone, Debug)]
pub struct TrainingArtifacts {
    pub model_path: PathBuf,
    pub labels_path: PathBuf,
}

#[derive(Default)]
struct TrainStats {
    loss_sum: f32,
    batches: usize,
    examples: usize,
    correct: usize,
}

impl TrainStats {
    fn record_loss(&mut self, loss: f32) {
        self.loss_sum += loss;
        self.batches += 1;
    }

    fn record_accuracy(&mut self, logits: &[i32], targets: &[i32]) {
        self.examples += targets.len();
        self.correct += logits
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();
    }

    fn avg_loss(&self) -> f32 {
        if self.batches == 0 {
            0.0
        } else {
            self.loss_sum / self.batches as f32
        }
    }

    fn accuracy(&self) -> f32 {
        if self.examples == 0 {
            0.0
        } else {
            self.correct as f32 / self.examples as f32
        }
    }
}

/// 主训练流程。
pub fn train(config: &TrainConfig) -> Result<TrainingArtifacts> {
    let dataset = PageDataset::from_config(&config.dataset)?;
    let (train_dataset, val_dataset) = dataset.split(config.val_split, config.seed);
    if train_dataset.len() == 0 {
        anyhow::bail!("训练集为空，请检查 manifest 或 split 配置");
    }

    let device = <TrainBackend as Backend>::Device::default();
    init_backend(&device);
    let train_len = train_dataset.len();
    let val_len = val_dataset.len();
    let (height, width) = train_dataset.image_size();
    let num_classes = train_dataset.num_classes();

    println!(
        "准备训练：训练样本 {}，验证样本 {}，类别数 {}，输入尺寸 {}x{}",
        train_len, val_len, num_classes, width, height
    );

    let train_loader = build_loader::<TrainBackend>(
        &train_dataset,
        config.batch_size,
        Some(config.seed),
        device.clone(),
    );

    let val_loader = if val_dataset.len() > 0 {
        Some(build_loader::<TrainBackend>(
            &val_dataset,
            config.batch_size,
            None,
            device.clone(),
        ))
    } else {
        None
    };
    let mut model =
        PageNet::<TrainBackend>::new(&device, height as usize, width as usize, num_classes);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();

    let total_items = train_loader.num_items();

    for epoch in 0..config.epochs {
        let mut stats = TrainStats::default();
        let mut loader_iter = train_loader.iter();
        let mut processed = 0usize;

        println!(
            "\n开始第 {}/{} 轮训练（批量大小 {}）",
            epoch + 1,
            config.epochs,
            config.batch_size
        );

        while let Some(batch) = loader_iter.next() {
            let logits = model.forward(batch.images.clone());
            let loss = loss_fn.forward(logits.clone(), batch.targets.clone());
            let loss_value = loss.clone().into_scalar();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);

            stats.record_loss(loss_value);
            processed += batch.targets.dims()[0];
            print!(
                "\r  训练进度 {:>6}/{:<6} 样本 | 当前批次损失 {:.4}",
                processed, total_items, loss_value
            );
            let _ = io::stdout().flush();
        }
        println!();

        print!(
            "epoch {:>3}/{:<3} | train_loss: {:.4}",
            epoch + 1,
            config.epochs,
            stats.avg_loss()
        );

        if let Some(val_loader) = &val_loader {
            println!("\n  开始验证……");
            match evaluate(&model, &loss_fn, val_loader) {
                Ok(eval) => {
                    print!(
                        " | val_loss: {:.4} | val_acc: {:>6.2}%",
                        eval.avg_loss(),
                        eval.accuracy() * 100.0
                    );
                }
                Err(err) => {
                    eprintln!("计算验证指标失败：{err}");
                    print!(" | val_metrics: error");
                }
            }
        }
        println!();
    }

    // 推理模型（去掉自动求导），并保存。
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("无法创建输出目录 {}", config.output_dir.display()))?;

    let model_path = config.output_dir.join("page_net");
    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .valid()
        .save_file(&model_path, &recorder)
        .with_context(|| format!("保存模型失败：{}", model_path.display()))?;

    let labels_path = config.output_dir.join("labels.json");
    write_labels(&labels_path, dataset.label_names())?;

    Ok(TrainingArtifacts {
        model_path,
        labels_path,
    })
}

fn build_loader<B: Backend>(
    dataset: &PageDataset,
    batch_size: usize,
    shuffle_seed: Option<u64>,
    device: B::Device,
) -> Arc<dyn DataLoader<B, PageBatch<B>>> {
    let batcher = dataset.batcher::<B>();
    let builder = DataLoaderBuilder::<B, PageSample, PageBatch<B>>::new(batcher)
        .batch_size(batch_size)
        .set_device(device);

    let builder = if let Some(seed) = shuffle_seed {
        builder.shuffle(seed)
    } else {
        builder
    };

    builder.build(dataset.clone())
}

fn evaluate(
    model: &PageNet<TrainBackend>,
    loss_fn: &CrossEntropyLoss<TrainBackend>,
    loader: &Arc<dyn DataLoader<TrainBackend, PageBatch<TrainBackend>>>,
) -> Result<TrainStats> {
    let mut stats = TrainStats::default();
    let mut iter = loader.iter();

    while let Some(batch) = iter.next() {
        let logits = model.forward(batch.images.clone());
        let loss = loss_fn.forward(logits.clone(), batch.targets.clone());
        stats.record_loss(loss.into_scalar());

        let preds = logits
            .argmax(1)
            .inner()
            .into_data()
            .into_vec::<i32>()
            .map_err(|err| anyhow!("读取预测张量失败：{err:?}"))?;
        let targets = batch
            .targets
            .clone()
            .inner()
            .into_data()
            .into_vec::<i32>()
            .map_err(|err| anyhow!("读取标签张量失败：{err:?}"))?;
        stats.record_accuracy(&preds, &targets);
    }

    Ok(stats)
}

fn write_labels(path: &Path, labels: Arc<Vec<String>>) -> Result<()> {
    let parent = path.parent();
    if let Some(dir) = parent {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)
                .with_context(|| format!("无法创建标签文件目录 {}", dir.display()))?;
        }
    }
    let file =
        File::create(path).with_context(|| format!("无法创建标签文件 {}", path.display()))?;
    serde_json::to_writer_pretty(file, &*labels)
        .with_context(|| format!("写入标签文件失败 {}", path.display()))
}
