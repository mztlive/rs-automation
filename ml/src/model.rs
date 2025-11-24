use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::Conv2dConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{Tensor, activation::relu, backend::Backend},
};

/// 简单的 CNN，用于页面分类。
#[derive(Module, Debug)]
pub struct PageNet<B: Backend> {
    conv1: burn::nn::conv::Conv2d<B>,
    conv2: burn::nn::conv::Conv2d<B>,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    fc1: Linear<B>,
    fc_out: Linear<B>,
}

impl<B: Backend> PageNet<B> {
    /// 创建网络。
    ///
    /// # 参数
    /// - `device`: 设备（对 NdArray 后端即 CPU）。
    /// - `input_height`: 归一化后图像高度。
    /// - `input_width`: 归一化后图像宽度。
    /// - `num_classes`: 类别数量。
    pub fn new(
        device: &B::Device,
        input_height: usize,
        input_width: usize,
        num_classes: usize,
    ) -> Self {
        assert!(
            input_height >= 4 && input_width >= 4,
            "输入尺寸至少需要 4x4，当前为 {}x{}",
            input_height,
            input_width
        );

        let conv1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let pool1 = MaxPool2dConfig::new([2, 2]).init();
        let pool2 = MaxPool2dConfig::new([2, 2]).init();

        let height_after = (input_height / 2).max(1) / 2;
        let width_after = (input_width / 2).max(1) / 2;
        let flattened = 32 * height_after * width_after;

        let fc1 = LinearConfig::new(flattened, 128).init(device);
        let fc_out = LinearConfig::new(128, num_classes).init(device);

        Self {
            conv1,
            conv2,
            pool1,
            pool2,
            fc1,
            fc_out,
        }
    }

    /// 前向推理，返回 logits。
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(input));
        let x = self.pool1.forward(x);

        let x = relu(self.conv2.forward(x));
        let x = self.pool2.forward(x);

        let x = x.flatten(1, 3);
        let x = relu(self.fc1.forward(x));
        self.fc_out.forward(x)
    }
}
