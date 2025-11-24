use opencv::core::{
    self, AlgorithmHint, BORDER_REPLICATE, Mat, MatTraitConst, Point2f, Rect, Scalar, Size, Vec3b,
    copy_make_border,
};
use opencv::imgproc;
use opencv::prelude::MatExprTraitConst;

/// 描述几何/滤波操作在触碰图像边缘时的填充策略。
#[derive(Clone, Copy)]
pub struct BorderOptions {
    /// OpenCV 边缘模式枚举值（如 `core::BORDER_CONSTANT`、`core::BORDER_REPLICATE`）。
    pub mode: i32,
    /// 当使用常量填充时写入的 RGBA 颜色；对复制等模式会被忽略。
    pub value: Scalar,
}

impl Default for BorderOptions {
    fn default() -> Self {
        Self {
            mode: BORDER_REPLICATE,
            value: Scalar::default(),
        }
    }
}

/// 对输入图像执行亚像素级平移，同时根据 `BorderOptions` 填充外露区域。
///
/// # 参数
/// - `image`: 源图像矩阵，通常为 BGR 通道。
/// - `shift_x`: 水平方向平移像素数，正值向右，支持小数。
/// - `shift_y`: 垂直方向平移像素数，正值向下，支持小数。
/// - `border`: 边缘填充策略，用于补齐平移后出现的空白区域。
///
/// # 返回
/// - 平移后的新 `Mat`，尺寸与输入一致。
///
/// # 错误
/// - 当输入矩阵尺寸非法或 OpenCV 仿射变换失败时返回 `opencv::Error`。
pub fn translate(
    image: &Mat,
    shift_x: f64,
    shift_y: f64,
    border: BorderOptions,
) -> opencv::Result<Mat> {
    let size = image.size()?;
    let transform = Mat::from_slice_2d(&[[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])?;
    let mut output = Mat::default();
    imgproc::warp_affine(
        image,
        &mut output,
        &transform,
        size,
        imgproc::INTER_LINEAR,
        border.mode,
        border.value,
    )?;
    Ok(output)
}

/// 以等比缩放图像并保持目标尺寸恒定，必要时对周边做居中裁剪或补边。
///
/// 该方法适合模拟轻微缩放模糊：放大超出原尺寸时会居中裁切；缩小则将图像贴回原画布并使用 `border`
/// 进行填充，以保证输出大小与输入一致。
///
/// # 参数
/// - `image`: 源图像矩阵。
/// - `scale`: 缩放比例，>1 表示放大，<1 表示缩小。
/// - `interpolation`: OpenCV 插值策略（如 `imgproc::INTER_CUBIC`）。
/// - `border`: 当缩小时若出现空白，用于补齐区域的边缘策略。
pub fn scale_preserve(
    image: &Mat,
    scale: f64,
    interpolation: i32,
    border: BorderOptions,
) -> opencv::Result<Mat> {
    let size = image.size()?;
    let width = size.width;
    let height = size.height;
    let new_width = ((width as f64) * scale).round().max(1.0) as i32;
    let new_height = ((height as f64) * scale).round().max(1.0) as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        image,
        &mut resized,
        Size::new(new_width, new_height),
        0.0,
        0.0,
        interpolation,
    )?;

    if new_width == width && new_height == height {
        return Ok(resized);
    }

    if new_width >= width && new_height >= height {
        let offset_x = ((new_width - width) / 2).max(0);
        let offset_y = ((new_height - height) / 2).max(0);
        let roi = Rect::new(offset_x, offset_y, width, height);
        let cropped = Mat::roi(&resized, roi)?;
        return cropped.try_clone();
    }

    let left = ((width - new_width) / 2).max(0);
    let right = width - new_width - left;
    let top = ((height - new_height) / 2).max(0);
    let bottom = height - new_height - top;
    let mut padded = Mat::default();
    copy_make_border(
        &resized,
        &mut padded,
        top,
        bottom,
        left,
        right,
        border.mode,
        border.value,
    )?;
    let mut result = padded;
    let padded_size = result.size()?;
    if padded_size.width != width || padded_size.height != height {
        let offset_x = ((padded_size.width - width) / 2).max(0);
        let offset_y = ((padded_size.height - height) / 2).max(0);
        let available_width = (padded_size.width - offset_x).max(1);
        let available_height = (padded_size.height - offset_y).max(1);
        let crop_width = width.min(available_width).max(1);
        let crop_height = height.min(available_height).max(1);
        let roi = Rect::new(offset_x, offset_y, crop_width, crop_height);
        let cropped = Mat::roi(&result, roi)?;
        result = cropped.try_clone()?;
    }
    Ok(result)
}

/// 绕图像中心进行旋转（单位：度），并根据指定边缘策略填充空白。
///
/// # 参数
/// - `image`: 源图像矩阵。
/// - `angle_deg`: 顺时针角度（度），小角度适合模拟轻微倾斜。
/// - `border`: 用于填充旋转后外露区域的策略。
pub fn rotate(image: &Mat, angle_deg: f64, border: BorderOptions) -> opencv::Result<Mat> {
    let size = image.size()?;
    let center = Point2f::new((size.width as f32) / 2.0, (size.height as f32) / 2.0);
    let rotation = imgproc::get_rotation_matrix_2d(center, angle_deg, 1.0)?;
    let mut output = Mat::default();
    imgproc::warp_affine(
        image,
        &mut output,
        &rotation,
        size,
        imgproc::INTER_LINEAR,
        border.mode,
        border.value,
    )?;
    Ok(output)
}

/// 按亮度、对比度与饱和度三个维度对图像做细粒度扰动。
///
/// - 先使用 `alpha`、`beta` 调整亮度/对比度；
/// - 若 `saturation` 不等于 1，则转到 HSV 空间修改饱和度通道；
/// - 处理过程保持尺寸与通道不变。
///
/// # 参数
/// - `image`: 源 BGR 图像。
/// - `alpha`: 乘法增益（对比度）。
/// - `beta`: 加法偏移（亮度）。
/// - `saturation`: 对饱和度的倍乘因子，1 为保持不变。
pub fn adjust_color(image: &Mat, alpha: f64, beta: f64, saturation: f64) -> opencv::Result<Mat> {
    let mut adjusted = Mat::default();
    image.convert_to(&mut adjusted, image.typ(), alpha, beta)?;

    if (saturation - 1.0).abs() < f64::EPSILON {
        return Ok(adjusted);
    }

    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &adjusted,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut channels = core::Vector::<Mat>::new();
    core::split(&hsv, &mut channels)?;

    if channels.len() > 1 {
        let sat = channels.get(1)?.try_clone()?;
        let mut sat_float = Mat::default();
        sat.convert_to(&mut sat_float, core::CV_32F, 1.0, 0.0)?;
        let factor = Mat::new_rows_cols_with_default(
            sat_float.rows(),
            sat_float.cols(),
            core::CV_32F,
            Scalar::all(saturation),
        )?;
        let mut scaled = Mat::default();
        core::multiply(&sat_float, &factor, &mut scaled, 1.0, -1)?;
        let max_mat = Mat::new_rows_cols_with_default(
            sat_float.rows(),
            sat_float.cols(),
            core::CV_32F,
            Scalar::all(255.0),
        )?;
        let mut clamped_high = Mat::default();
        core::min(&scaled, &max_mat, &mut clamped_high)?;
        let zero_mat = Mat::zeros(sat_float.rows(), sat_float.cols(), core::CV_32F)?.to_mat()?;
        let mut clamped = Mat::default();
        core::max(&clamped_high, &zero_mat, &mut clamped)?;
        let mut sat_u8 = Mat::default();
        clamped.convert_to(&mut sat_u8, core::CV_8U, 1.0, 0.0)?;
        channels.set(1, sat_u8)?;
        core::merge(&channels, &mut hsv)?;
        imgproc::cvt_color(
            &hsv,
            &mut adjusted,
            imgproc::COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    }

    Ok(adjusted)
}

/// 在图像上叠加高斯噪声，模拟截图压缩或传感器噪声。
///
/// # 参数
/// - `image`: 源图像。
/// - `mean`: 噪声均值，建议 0。
/// - `stddev`: 噪声标准差，值越大波动越强。
///
/// # 返回
/// - 写入噪声后的新矩阵（与输入通道、尺寸一致）。
pub fn add_gaussian_noise(image: &Mat, mean: f64, stddev: f64) -> opencv::Result<Mat> {
    let mut float = Mat::default();
    image.convert_to(&mut float, core::CV_32F, 1.0, 0.0)?;
    let mut noise = Mat::zeros(float.rows(), float.cols(), float.typ())?.to_mat()?;
    core::randn(&mut noise, &Scalar::all(mean), &Scalar::all(stddev))?;
    let mut combined = Mat::default();
    core::add(&float, &noise, &mut combined, &core::no_array(), -1)?;
    let mut output = Mat::default();
    combined.convert_to(&mut output, image.typ(), 1.0, 0.0)?;
    Ok(output)
}

/// 对图像应用高斯模糊，以模拟运动模糊或失焦。
///
/// # 参数
/// - `image`: 源图像。
/// - `sigma`: 高斯核的标准差，值越大模糊越明显。
/// - `border`: 模糊时的边缘填充策略。
pub fn gaussian_blur(image: &Mat, sigma: f64, border: BorderOptions) -> opencv::Result<Mat> {
    let kernel = ((sigma * 6.0).ceil() as i32 | 1).max(3);
    let size = Size::new(kernel, kernel);
    let mut output = Mat::default();
    imgproc::gaussian_blur(
        image,
        &mut output,
        size,
        sigma,
        sigma,
        border.mode,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(output)
}

/// 在指定区域内绘制不透明矩形遮挡，可模拟通知弹窗或鼠标遮挡。
///
/// # 参数
/// - `image`: 源图像。
/// - `rect`: 遮挡的像素矩形区域。
/// - `color`: 遮挡色（BGR24），透明度默认 255。
pub fn occlude_rect(image: &Mat, rect: Rect, color: Scalar) -> opencv::Result<Mat> {
    let mut output = image.try_clone()?;
    imgproc::rectangle(&mut output, rect, color, -1, imgproc::LINE_8, 0)?;
    Ok(output)
}

/// 根据图像四周像素估算背景色，常用于选择常量填充色。
///
/// # 参数
/// - `image`: 源图像。
///
/// # 返回
/// - 由四条边像素平均值推算出的 `Scalar(B,G,R,A)`。
pub fn estimate_edge_color(image: &Mat) -> opencv::Result<Scalar> {
    let size = image.size()?;
    let width = size.width;
    let height = size.height;
    if width <= 0 || height <= 0 {
        return Ok(Scalar::all(0.0));
    }

    let mut sum = [0u64; 3];
    let mut count = 0u64;

    for x in 0..width {
        let top = *image.at_2d::<Vec3b>(0, x)?;
        let bottom = *image.at_2d::<Vec3b>(height - 1, x)?;
        sum[0] += top[0] as u64;
        sum[1] += top[1] as u64;
        sum[2] += top[2] as u64;
        sum[0] += bottom[0] as u64;
        sum[1] += bottom[1] as u64;
        sum[2] += bottom[2] as u64;
        count += 2;
    }

    for y in 1..(height - 1) {
        let left = *image.at_2d::<Vec3b>(y, 0)?;
        let right = *image.at_2d::<Vec3b>(y, width - 1)?;
        sum[0] += left[0] as u64;
        sum[1] += left[1] as u64;
        sum[2] += left[2] as u64;
        sum[0] += right[0] as u64;
        sum[1] += right[1] as u64;
        sum[2] += right[2] as u64;
        count += 2;
    }

    if count == 0 {
        return Ok(Scalar::all(0.0));
    }

    Ok(Scalar::new(
        sum[0] as f64 / count as f64,
        sum[1] as f64 / count as f64,
        sum[2] as f64 / count as f64,
        255.0,
    ))
}
