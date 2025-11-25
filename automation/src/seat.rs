//! 座位数据结构与检测/聚类算法，供 LocateSeat 等步骤复用。

use anyhow::{Result, anyhow};
use opencv::{
    core::{self, Mat, Point, Scalar, Size},
    imgproc,
    prelude::*,
};

/// 座位状态：可选/已选/不可选/未知。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatState {
    Available,
    Selected,
    Unavailable,
    Unknown,
}

/// 座位描述，包含行列、状态以及图像/屏幕坐标。
#[derive(Debug, Clone)]
pub struct Seat {
    pub row: usize,
    pub col: usize,
    pub state: SeatState,
    pub bbox: core::Rect,
    pub center_img: (f64, f64),
    pub center_screen: (i32, i32),
}

impl Seat {
    /// 屏幕坐标（窗口左上角为基准）。
    pub fn screen_point(&self) -> (i32, i32) {
        self.center_screen
    }

    /// 图像坐标（原始截图坐标系）。
    pub fn image_point(&self) -> (f64, f64) {
        self.center_img
    }

    /// 是否可选（灰色）。
    pub fn is_available(&self) -> bool {
        matches!(self.state, SeatState::Available)
    }

    /// 是否已选（绿色）。
    pub fn is_selected(&self) -> bool {
        matches!(self.state, SeatState::Selected)
    }

    /// 是否不可选（红色）。
    pub fn is_unavailable(&self) -> bool {
        matches!(self.state, SeatState::Unavailable)
    }
}

/// 座位区域，按行存储的二维数组。
#[derive(Debug, Clone)]
pub struct SeatArea {
    grid: Vec<Vec<Seat>>,
}

impl SeatArea {
    /// 返回行列表。
    pub fn rows(&self) -> &[Vec<Seat>] {
        &self.grid
    }

    /// 行数（1-based 最大行号）。
    pub fn row_count(&self) -> usize {
        self.grid.len()
    }

    /// 获取指定行的座位切片（1-based）。
    pub fn row(&self, row: usize) -> Option<&[Seat]> {
        self.grid.get(row.saturating_sub(1)).map(|r| r.as_slice())
    }

    /// 获取指定行的列数（1-based 行号）。
    pub fn cols_in_row(&self, row: usize) -> Option<usize> {
        self.grid.get(row.saturating_sub(1)).map(|r| r.len())
    }

    /// 按行列索引座位（1-based）。
    pub fn seat(&self, row: usize, col: usize) -> Option<&Seat> {
        self.grid
            .get(row.saturating_sub(1))
            .and_then(|r| r.get(col.saturating_sub(1)))
    }

    /// 扁平迭代所有座位。
    pub fn iter(&self) -> impl Iterator<Item = &Seat> {
        self.grid.iter().flat_map(|r| r.iter())
    }
}

/// ROI 预处理结果，包含 HSV 矩阵及坐标信息。
pub struct RoiContext {
    pub hsv: Mat,
    pub roi_origin: (u32, u32),
    pub img_dims: (u32, u32),
}

/// 裁剪 ROI 并返回 ROI BGR、原图尺寸和 ROI 左上角偏移。
///
/// - 通过 `resolve_roi` 校验/默认 ROI
/// - 使用 `Mat::roi` 抽取子图并拷贝
/// - 返回 `(roi_bgr, roi_origin, img_dims)`，便于后续转换或保存
pub fn crop_bgr_with_roi(
    bgr: &Mat,
    roi: Option<(u32, u32, u32, u32)>,
) -> Result<(Mat, (u32, u32), (u32, u32))> {
    let img_size = bgr.size()?;
    let (img_w, img_h) = (img_size.width as u32, img_size.height as u32);
    let (roi_x, roi_y, roi_w, roi_h) = resolve_roi((img_w, img_h), roi)?;

    let rect = core::Rect::new(roi_x as i32, roi_y as i32, roi_w as i32, roi_h as i32);
    let roi_bgr = bgr.roi(rect)?.try_clone()?;
    Ok((roi_bgr, (roi_x, roi_y), (img_w, img_h)))
}

/// 将 BGR 矩阵转换为 HSV。
///
/// - 使用 OpenCV `cvt_color`，色彩空间 BGR -> HSV
/// - 返回新的 Mat，不修改输入
pub fn bgr_to_hsv(bgr: &Mat) -> Result<Mat> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        bgr,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(hsv)
}

/// 将 BGR 图像转换为 ROI 范围内的 HSV，并返回 ROI 坐标信息。
///
/// - 使用 `crop_bgr_with_roi` 获取 ROI BGR、偏移与原图尺寸
/// - 调用 `bgr_to_hsv` 得到 HSV 数据
/// - 返回 HSV 数据、ROI 左上角偏移以及原图尺寸，便于后续坐标换算
pub fn roi_hsv_from_bgr(bgr: &Mat, roi: Option<(u32, u32, u32, u32)>) -> Result<RoiContext> {
    let (roi_bgr, roi_origin, img_dims) = crop_bgr_with_roi(bgr, roi)?;
    let hsv = bgr_to_hsv(&roi_bgr)?;

    Ok(RoiContext {
        hsv,
        roi_origin,
        img_dims,
    })
}

/// 解析 ROI，若未指定则使用整张图；并校验范围。
///
/// - 默认 ROI 为整张图 `(0, 0, img_w, img_h)`
/// - ROI 超界立即返回错误，避免裁剪阶段崩溃
pub fn resolve_roi(
    img_dims: (u32, u32),
    roi: Option<(u32, u32, u32, u32)>,
) -> Result<(u32, u32, u32, u32)> {
    let (img_w, img_h) = img_dims;
    let (roi_x, roi_y, roi_w, roi_h) = roi.unwrap_or((0, 0, img_w, img_h));
    if roi_x + roi_w > img_w || roi_y + roi_h > img_h {
        return Err(anyhow!("ROI 超出截图范围"));
    }
    Ok((roi_x, roi_y, roi_w, roi_h))
}

/// 创建颜色掩膜并闭运算去噪。
///
/// - 遍历 `seat_color_ranges` 生成掩膜并 OR 合并
/// - 使用 3x3 矩形核做闭运算，填平孔洞并清理小噪点
pub fn build_mask(hsv: &Mat) -> Result<Mat> {
    let mut mask = Mat::zeros(hsv.rows(), hsv.cols(), core::CV_8UC1)?.to_mat()?;
    for (lower, upper) in seat_color_ranges() {
        let mut m = Mat::default();
        core::in_range(hsv, &lower, &upper, &mut m)?;
        let mut merged = Mat::default();
        core::bitwise_or(&mask, &m, &mut merged, &core::no_array())?;
        mask = merged;
    }

    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;

    let mut cleaned = Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut cleaned,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(cleaned)
}

/// 将轮廓转换为方块并按尺寸过滤。
///
/// - `find_contours` + `bounding_rect` 取外接矩形
/// - 仅保留宽高 12–80px 的方块，减少 legend/按钮干扰（适配更大座位块）
/// - 无有效方块时返回错误，提示调整 ROI/颜色范围
pub fn contours_to_boxes(mask: &Mat) -> Result<Vec<core::Rect>> {
    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(
        mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut boxes: Vec<core::Rect> = Vec::new();
    for c in contours.iter() {
        let rect = imgproc::bounding_rect(&c)?;
        if rect.width >= 12 && rect.width <= 80 && rect.height >= 12 && rect.height <= 80 {
            boxes.push(rect);
        }
    }
    if boxes.is_empty() {
        return Err(anyhow!("未检测到座位方块，请调整颜色范围或 ROI"));
    }
    Ok(boxes)
}

/// 按 y 聚类为行。
///
/// - 以方块中心 y 与行首元素比较，偏差在 `row_tolerance` 内则归为同一行
/// - 处理完全部方块后若行列表为空，返回错误
pub fn cluster_rows(boxes: Vec<core::Rect>, row_tolerance: i32) -> Result<Vec<Vec<core::Rect>>> {
    let mut rows: Vec<Vec<core::Rect>> = Vec::new();
    for rect in boxes {
        let cy = rect.y + rect.height / 2;
        if let Some(row) = rows.iter_mut().find(|r| {
            let ref_rect = r.first().unwrap();
            let ref_y = ref_rect.y + ref_rect.height / 2;
            (ref_y - cy).abs() <= row_tolerance
        }) {
            row.push(rect);
        } else {
            rows.push(vec![rect]);
        }
    }
    if rows.is_empty() {
        return Err(anyhow!("未能聚类出行数据"));
    }
    Ok(rows)
}

/// 对行/列排序，确保从上到下、从左到右。
///
/// - 行按最小 y 升序
/// - 每行内按 x 升序
pub fn sort_rows_and_cols(rows: &mut [Vec<core::Rect>]) {
    rows.sort_by_key(|r| r.iter().map(|x| x.y).min().unwrap_or(0));
    for row in rows.iter_mut() {
        row.sort_by_key(|r| r.x);
    }
}

/// 计算图像->窗口缩放比例。
///
/// - 返回 `(scale_x, scale_y)`，用于将截图坐标换算到窗口逻辑坐标
/// - 若窗口尺寸与截图一致，结果约为 (1.0, 1.0)
pub fn compute_scale_factors(img_dims: (u32, u32), window_size: (u32, u32)) -> (f32, f32) {
    (
        img_dims.0 as f32 / window_size.0 as f32,
        img_dims.1 as f32 / window_size.1 as f32,
    )
}

/// 从聚类结果构建座位列表，包含行/列/状态及坐标。
///
/// - 行号、列号从 1 开始，按 `sort_rows_and_cols` 后的顺序赋值
/// - 屏幕坐标基于 ROI 偏移、窗口位置与缩放比计算
/// - 状态通过 ROI HSV 图的中心像素判断
pub fn build_seats(
    rows: &[Vec<core::Rect>],
    roi_origin: (u32, u32),
    window_pos: (i32, i32),
    scale: (f32, f32),
    hsv_roi: &Mat,
) -> Vec<Seat> {
    let mut seats = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, rect) in row.iter().enumerate() {
            let center_px = rect_center(rect, (roi_origin.0 as i32, roi_origin.1 as i32));
            let logical_x = center_px.0 as f32 / scale.0;
            let logical_y = center_px.1 as f32 / scale.1;
            let screen_x = window_pos.0 + logical_x.round() as i32;
            let screen_y = window_pos.1 + logical_y.round() as i32;

            seats.push(Seat {
                row: row_idx + 1,
                col: col_idx + 1,
                state: seat_state_from_hsv(hsv_roi, rect),
                bbox: *rect,
                center_img: center_px,
                center_screen: (screen_x, screen_y),
            });
        }
    }
    seats
}

/// 构建 SeatArea（二维数组），便于按行列索引。
pub fn build_seat_area(
    rows: &[Vec<core::Rect>],
    roi_origin: (u32, u32),
    window_pos: (i32, i32),
    scale: (f32, f32),
    hsv_roi: &Mat,
) -> SeatArea {
    let seats = build_seats(rows, roi_origin, window_pos, scale, hsv_roi);
    let mut grid: Vec<Vec<Seat>> = Vec::new();
    for seat in seats {
        if grid.len() < seat.row {
            grid.resize(seat.row, Vec::new());
        }
        grid[seat.row - 1].push(seat);
    }
    SeatArea { grid }
}

/// 将状态枚举转成可读文本。
pub fn format_seat_state(state: SeatState) -> &'static str {
    match state {
        SeatState::Available => "可选",
        SeatState::Selected => "已选",
        SeatState::Unavailable => "不可选",
        SeatState::Unknown => "未知",
    }
}

/// 取指定排/列的屏幕坐标，并返回图像中心坐标。
///
/// - 检查列号范围，非法时返回错误
/// - 在 ROI 坐标系计算矩形中心，再按缩放比例换算逻辑坐标
/// - 将窗口左上角偏移加到逻辑坐标，得到最终屏幕位置
pub fn seat_screen_point(
    seats: &[core::Rect],
    col: usize,
    target_row: usize,
    roi_origin: (u32, u32),
    window_pos: (i32, i32),
    scale: (f32, f32),
) -> Result<(i32, i32, (f64, f64))> {
    if col == 0 || col > seats.len() {
        return Err(anyhow!(
            "目标座位超出范围：第 {} 排没有第 {} 座（本排 {} 个）",
            target_row,
            col,
            seats.len()
        ));
    }
    let rect = seats[col - 1];
    let center_px = rect_center(&rect, (roi_origin.0 as i32, roi_origin.1 as i32));
    let logical_x = center_px.0 as f32 / scale.0;
    let logical_y = center_px.1 as f32 / scale.1;
    let screen_x = window_pos.0 + logical_x.round() as i32;
    let screen_y = window_pos.1 + logical_y.round() as i32;
    Ok((screen_x, screen_y, center_px))
}

/// 座位颜色的 HSV 范围列表：灰/绿/红。
///
/// - 灰：低饱和度，可选座位
/// - 绿：已选座位
/// - 红：不可选座位（包含低/高 Hue 两段以覆盖环绕区间）
pub fn seat_color_ranges() -> Vec<(Scalar, Scalar)> {
    vec![
        // 灰色（可选）：低饱和度
        (
            Scalar::new(0.0, 0.0, 60.0, 0.0),
            Scalar::new(180.0, 60.0, 210.0, 0.0),
        ),
        // 绿色（已选）
        (
            Scalar::new(60.0, 80.0, 120.0, 0.0),
            Scalar::new(90.0, 255.0, 255.0, 0.0),
        ),
        // 红色（不可选）：包含低端与高端 hue
        (
            Scalar::new(0.0, 80.0, 120.0, 0.0),
            Scalar::new(15.0, 255.0, 255.0, 0.0),
        ),
        (
            Scalar::new(165.0, 80.0, 120.0, 0.0),
            Scalar::new(180.0, 255.0, 255.0, 0.0),
        ),
    ]
}

/// 按状态拆分的 HSV 范围列表，便于分类。
pub fn seat_state_color_ranges() -> Vec<(SeatState, (Scalar, Scalar))> {
    vec![
        (
            SeatState::Available,
            (
                Scalar::new(0.0, 0.0, 60.0, 0.0),
                Scalar::new(180.0, 60.0, 210.0, 0.0),
            ),
        ),
        (
            SeatState::Selected,
            (
                Scalar::new(60.0, 80.0, 120.0, 0.0),
                Scalar::new(90.0, 255.0, 255.0, 0.0),
            ),
        ),
        (
            SeatState::Unavailable,
            (
                Scalar::new(0.0, 80.0, 120.0, 0.0),
                Scalar::new(15.0, 255.0, 255.0, 0.0),
            ),
        ),
        (
            SeatState::Unavailable,
            (
                Scalar::new(165.0, 80.0, 120.0, 0.0),
                Scalar::new(180.0, 255.0, 255.0, 0.0),
            ),
        ),
    ]
}

/// 根据 ROI HSV 的中心像素推断座位状态。
///
/// - 中心点越界或读值失败时返回 `Unknown`
/// - 匹配灰/绿/红范围后返回对应状态，否则 `Unknown`
pub fn seat_state_from_hsv(hsv_roi: &Mat, rect: &core::Rect) -> SeatState {
    let cx = rect.x + rect.width / 2;
    let cy = rect.y + rect.height / 2;
    if cx < 0 || cy < 0 || cx >= hsv_roi.cols() || cy >= hsv_roi.rows() {
        return SeatState::Unknown;
    }
    let pixel = match hsv_roi.at_2d::<core::Vec3b>(cy, cx) {
        Ok(p) => p,
        Err(_) => return SeatState::Unknown,
    };
    let scalar = Scalar::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64, 0.0);
    for (state, (lower, upper)) in seat_state_color_ranges() {
        if scalar_in_range(&scalar, &lower, &upper) {
            return state;
        }
    }
    SeatState::Unknown
}

/// 判断单个 HSV 像素是否位于给定范围。
pub fn scalar_in_range(val: &Scalar, lower: &Scalar, upper: &Scalar) -> bool {
    (val[0] >= lower[0] && val[0] <= upper[0])
        && (val[1] >= lower[1] && val[1] <= upper[1])
        && (val[2] >= lower[2] && val[2] <= upper[2])
}

/// 计算矩形的中心（加上 ROI 偏移）。
///
/// - 以矩形左上角和宽高求中心
/// - 将 ROI 偏移叠加，返回全图坐标系下的中心点
pub fn rect_center(rect: &core::Rect, offset: (i32, i32)) -> (f64, f64) {
    (
        offset.0 as f64 + rect.x as f64 + rect.width as f64 * 0.5,
        offset.1 as f64 + rect.y as f64 + rect.height as f64 * 0.5,
    )
}

/// 求掩膜中非零像素的最小外接矩形，返回相对掩膜的 Rect。
///
/// - 掩膜全零时返回 `Ok(None)`
/// - 否则使用 `bounding_rect` 包裹所有非零点
pub fn mask_nonzero_bbox(mask: &Mat) -> Result<Option<core::Rect>> {
    let mut points = core::Vector::<core::Point>::new();
    core::find_non_zero(mask, &mut points)?;
    if points.is_empty() {
        return Ok(None);
    }
    let rect = imgproc::bounding_rect(&points)?;
    Ok(Some(rect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use opencv::core::CV_8UC3;
    use opencv::imgcodecs;
    use std::path::PathBuf;

    /// 构造纯色 BGR Mat，便于独立测试。
    ///
    /// - 指定宽、高和 BGR 颜色
    /// - 返回可直接用于 ROI/HVS 辅助函数
    fn solid_bgr(width: i32, height: i32, color: Scalar) -> Result<Mat> {
        Mat::new_size_with_default(Size::new(width, height), core::CV_8UC3, color)
            .map_err(anyhow::Error::from)
    }

    #[test]
    fn roi_hsv_from_bgr_respects_roi_and_dims() -> Result<()> {
        let bgr = solid_bgr(6, 4, Scalar::new(10.0, 20.0, 30.0, 0.0))?;
        let roi = Some((1, 1, 3, 2));

        let ctx = roi_hsv_from_bgr(&bgr, roi)?;
        let hsv_size = ctx.hsv.size()?;

        assert_eq!(ctx.img_dims, (6, 4));
        assert_eq!(ctx.roi_origin, (1, 1));
        assert_eq!((hsv_size.width, hsv_size.height), (3, 2));
        Ok(())
    }

    #[test]
    fn resolve_roi_defaults_to_full_image() -> Result<()> {
        let roi = resolve_roi((5, 4), None)?;
        assert_eq!(roi, (0, 0, 5, 4));
        Ok(())
    }

    #[test]
    fn resolve_roi_rejects_out_of_bounds() {
        let err = resolve_roi((10, 10), Some((9, 0, 2, 5))).unwrap_err();
        assert!(err.to_string().contains("ROI 超出截图范围"));
    }

    /// 从真实座位图中截取 ROI，验证尺寸与偏移输出。
    #[test]
    fn roi_hsv_from_bgr_real_image_roi_size_matches() -> Result<()> {
        let path: PathBuf =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/raw/movie-seats/1.png");
        let bgr = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        let size = bgr.size()?;
        assert!(
            size.width > 0 && size.height > 0,
            "image should not be empty"
        );

        // 该 ROI 覆盖座位区域，来自人工标定。
        // 缩进左侧以排除行号/标签，宽度收窄以减少伪检测。
        let roi = Some((50, 700, (size.width - 50) as u32, 530));

        let (roi_bgr, origin, dims) = crop_bgr_with_roi(&bgr, roi)?;
        let out_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/roi_hsv_from_bgr_real.png");
        imgcodecs::imwrite(out_path.to_str().unwrap(), &roi_bgr, &core::Vector::new())?;
        assert!(
            out_path.exists(),
            "ROI image should be written for inspection"
        );

        let ctx = roi_hsv_from_bgr(&bgr, roi)?;

        assert_eq!(dims, (size.width as u32, size.height as u32));
        assert_eq!(ctx.roi_origin, origin);
        assert_eq!(ctx.img_dims, dims);
        Ok(())
    }

    /// 座位状态按中心像素的 HSV 范围分类。
    #[test]
    fn seat_state_from_hsv_classifies_simple_colors() -> Result<()> {
        let available = Mat::new_size_with_default(
            Size::new(5, 5),
            CV_8UC3,
            Scalar::new(0.0, 0.0, 120.0, 0.0),
        )?;
        let rect = core::Rect::new(0, 0, 4, 4);
        assert_eq!(seat_state_from_hsv(&available, &rect), SeatState::Available);

        let selected = Mat::new_size_with_default(
            Size::new(5, 5),
            CV_8UC3,
            Scalar::new(70.0, 200.0, 200.0, 0.0),
        )?;
        assert_eq!(seat_state_from_hsv(&selected, &rect), SeatState::Selected);

        let unavailable = Mat::new_size_with_default(
            Size::new(5, 5),
            CV_8UC3,
            Scalar::new(5.0, 200.0, 200.0, 0.0),
        )?;
        assert_eq!(
            seat_state_from_hsv(&unavailable, &rect),
            SeatState::Unavailable
        );

        let unknown = Mat::new_size_with_default(
            Size::new(5, 5),
            CV_8UC3,
            Scalar::new(130.0, 200.0, 200.0, 0.0),
        )?;
        assert_eq!(seat_state_from_hsv(&unknown, &rect), SeatState::Unknown);
        Ok(())
    }

    /// 从聚类框生成座位区域并验证行列与坐标。
    #[test]
    fn build_seat_area_assigns_row_col_and_coordinates() -> Result<()> {
        let rows = vec![vec![
            core::Rect::new(0, 0, 10, 10),
            core::Rect::new(12, 0, 10, 10),
        ]];
        let hsv = Mat::new_size_with_default(
            Size::new(30, 20),
            CV_8UC3,
            Scalar::new(0.0, 0.0, 120.0, 0.0),
        )?;
        let area = build_seat_area(&rows, (0, 0), (0, 0), (1.0, 1.0), &hsv);

        assert_eq!(area.row_count(), 1);
        assert_eq!(area.cols_in_row(1), Some(2));
        let s1 = area.seat(1, 1).unwrap();
        let s2 = area.seat(1, 2).unwrap();
        assert_eq!((s1.row, s1.col), (1, 1));
        assert_eq!((s2.row, s2.col), (1, 2));
        assert_eq!(s1.center_img, (5.0, 5.0));
        assert_eq!(s2.center_img.0.round(), 17.0);
        assert!(area.iter().all(|s| s.is_available()));
        Ok(())
    }

    /// 基于真实座位图提取完整座位数据并对 SeatArea 断言（行/列数量大于 0）。
    #[test]
    fn build_seats_from_real_image_and_dump() -> Result<()> {
        let path: PathBuf =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/raw/movie-seats/1.png");
        let bgr = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        let size = bgr.size()?;
        assert!(
            size.width > 0 && size.height > 0,
            "image should not be empty"
        );

        // 该 ROI 覆盖座位区域，来自人工标定。
        let roi = Some((50, 700, (size.width - 50) as u32, 530));
        let roi_ctx = roi_hsv_from_bgr(&bgr, roi)?;
        assert_eq!(
            roi_ctx.roi_origin,
            (50, 700),
            "ROI origin should reflect crop"
        );

        let mask = build_mask(&roi_ctx.hsv)?;
        let boxes = contours_to_boxes(&mask)?;
        let row_tol = {
            let avg_h =
                boxes.iter().map(|r| r.height).sum::<i32>() as f32 / boxes.len().max(1) as f32;
            (avg_h.round() as i32 + 6).max(8)
        };
        let mut rows = cluster_rows(boxes, row_tol)?;
        sort_rows_and_cols(&mut rows);

        let scale = compute_scale_factors(roi_ctx.img_dims, roi_ctx.img_dims);
        let area = build_seat_area(&rows, roi_ctx.roi_origin, (0, 0), scale, &roi_ctx.hsv);
        assert!(area.row_count() > 0, "should detect at least 1 row");
        for r in 1..=area.row_count() {
            let row = area.row(r).expect("row should exist");
            assert!(!row.is_empty(), "row {} should have seats", r);
        }
        Ok(())
    }
}
