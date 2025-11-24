use anyhow::Result;
use clap::Parser;
use ml::{PageDatasetConfig, TrainConfig, train};
use std::path::PathBuf;

// • 这些参数直接决定训练过程使用的数据、输入尺寸和超参数，具体作用如下：

//   - --manifest <path>：指向由 image-pre-cli 生成的 CSV（至少包含 image,label 列）。训练时只会加载清单列出的样本，任何未记录的图像都会被忽略。
//   - --image-root <dir>：作为清单中相对路径的解析根目录。若清单写的是 movies/xxx.png，实测路径会拼成 <image-root>/movies/xxx.png。
//   - --output-dir <dir>：训练输出目录。脚本会创建目录并写入：
//       - page_net.*：模型权重（Burn 记录格式，可用于推理或继续训练）。
//       - labels.json：按类别编号排列的标签列表，供自动化/推理时查找 index→名称。
//   - --image-width / --image-height：所有图像会被 resize 到的目标尺寸（默认 224×224）。调整它会影响模型的输入张量形状和前后卷积层特征图大小，应与 image-pre-cli 中使用的尺寸保持一致。
//   - --batch-size：单个 mini-batch 的样本数。较大可提升吞吐，但会占用更多内存；过小则梯度噪声大、收敛慢。
//   - --epochs：遍历训练集的轮数。轮数越多越可能拟合充分，但超出一定程度可能过拟合。
//   - --learning-rate：Adam 优化器的基础学习率。值过大可能导致损失震荡，过小则收敛缓慢。
//   - --val-split：从 manifest 中划出用作验证集的比例（0–1）。例如 0.1 表示 10% 样本用于计算 val_loss/val_acc，剩余 90% 参与训练。设成 0 则不做验证。
//   - --seed：控制数据集划分、shuffle 等的随机种子，方便复现。只要 manifest 内容不变，相同 seed 会产生一致的 train/val 拆分。
//   - --normalize（布尔开关，默认启用）：开启后会把像素从 0–255 线性缩放到 0–1，便于稳定训练。若关闭，模型将直接接收 0–255 的数值。

//   运行命令时只要补齐 manifest / image-root / output-dir 的实际路径，其它参数可按需要调整。

#[derive(Parser, Debug)]
#[command(name = "ml-train", about = "训练页面分类模型（Burn + NdArray）")]
struct Args {
    /// manifest CSV 文件路径
    #[arg(long)]
    manifest: PathBuf,

    /// 图像根目录（manifest 中的 image 会在此目录下解析）
    #[arg(long)]
    image_root: PathBuf,

    /// 输出目录（模型与标签映射）
    #[arg(long)]
    output_dir: PathBuf,

    /// 归一化后的图像宽度
    #[arg(long, default_value_t = 256)]
    image_width: u32,

    /// 归一化后的图像高度
    #[arg(long, default_value_t = 480)]
    image_height: u32,

    /// batch 大小
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// 训练轮数
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// 学习率
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// 验证集比例 (0-1]
    #[arg(long, default_value_t = 0.1)]
    val_split: f32,

    /// 是否将像素归一化到 [0,1]
    #[arg(long, default_value_t = true)]
    normalize: bool,

    /// 随机种子
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dataset = PageDatasetConfig {
        manifest: args.manifest,
        image_root: args.image_root,
        width: args.image_width,
        height: args.image_height,
        normalize: args.normalize,
    };

    let config = TrainConfig {
        dataset,
        output_dir: args.output_dir,
        batch_size: args.batch_size,
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        val_split: args.val_split,
        seed: args.seed,
    };

    let artifacts = train(&config)?;
    println!(
        "训练完成，模型保存在：{}，标签映射：{}",
        artifacts.model_path.display(),
        artifacts.labels_path.display()
    );

    Ok(())
}
