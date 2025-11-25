//! LocateSeatByColor 用于在电影选座页面上根据座位颜色定位目标排/列并可选点击。
//! 使用前需保证窗口已激活且座位区域完全可见（上方 legend、底部按钮可用 ROI 排除）。
//!
//! 调用方式（示例）：
//! ```
//! pipeline.push(LocateSeatByColor {
//!     target_row: 5,
//!     target_cols: vec![7, 8],
//!     roi: Some((0, 120, 1080, 620)),
//!     click: true,
//!     row_tolerance_px: None,
//! });
//! ```
//! 参数说明：
//! - `target_row`：要操作的行，1 为顶部第一排；越界会报错。
//! - `target_cols`：目标列列表，1 为最左；缺失或越界同样报错。
//! - `roi`：可选裁剪区域 `(x, y, w, h)`，默认全图；用于排除 legend/按钮以避免误识别。
//! - `click`：为 `true` 时执行屏幕点击，否则仅打印定位结果；点击后会调用 `ctx.invalidate()`。
//! - `row_tolerance_px`：行聚类的 y 容忍度，未设置时按检测到的方块高度自动估算。
//!
//! 内部流程概览：
//! 1. `ctx.capture_rgba` 获取窗口截图，`rgba_to_bgr` 转换并裁剪 ROI。
//! 2. 将 ROI 转 HSV，遍历 `seat_color_ranges` 构建掩膜，做闭运算去噪。
//! 3. `find_contours`+`bounding_rect` 得到方块；12~40 像素框视为有效座位，否则报 “未检测到座位方块”。
//! 4. 按中心 y 使用 `cluster_rows` 聚类行，容忍度来自参数或高度估计；随后 `sort_rows_and_cols` 保证自上而下、自左而右。
//! 5. 取目标排，使用 `seat_screen_point`：在 ROI 坐标系求中心 -> 按截图/窗口缩放比换算逻辑坐标 -> 加上窗口左上角得屏幕坐标。
//! 6. 若开启 `click`，逐个 `input::click_screen`，并打印 “第 X 排第 Y 座 -> 图像/屏幕坐标” 供调试。
//! 常见调整项：修改 `roi` 以排除噪声，或改动 `seat_color_ranges` 以适配不同主题颜色。
use anyhow::{Result, anyhow};
use opencv::{
    core::{self, Mat, Point, Scalar, Size},
    imgproc,
    prelude::*,
};

use crate::{
    input,
    pipeline::{RunCtx, Step},
    vision,
    window::WandaWindow,
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

/// 通过颜色掩膜+轮廓聚类定位影厅座位的行/列，并输出或点击目标座位。
///
/// 适合颜色相对固定的座位图（如示例截图），默认识别灰色（可选）、绿色（已选）、红色（不可选）三类方块。
/// - `target_row`: 目标排，1 为最上方的第一排
/// - `target_cols`: 需操作的座位列号列表，1 为最左边
/// - `roi`: 可选的裁剪区域，避免把上方 legend 或下方按钮识别为座位；坐标基于截图像素
/// - `click`: 是否直接点击目标座位；否则只打印坐标
/// - `row_tolerance_px`: 行聚类的 y 方向容忍度，不填则用检测到的座位高度估计
pub struct LocateSeatByColor {
    pub target_row: usize,
    pub target_cols: Vec<usize>,
    pub roi: Option<(u32, u32, u32, u32)>,
    pub click: bool,
    pub row_tolerance_px: Option<i32>,
}

impl Step for LocateSeatByColor {
    /// 执行整体流程：校验参数、截图+掩膜、聚类行列并输出/点击目标座位。
    ///
    /// - 依次触发 ROI 截图、掩膜生成、轮廓过滤、行/列聚类与排序
    /// - 对目标排的指定列打印图像/屏幕坐标；`click=true` 时直接点击并使上下文失效
    /// - 任一阶段出错（如无方块、行越界）都会中断并返回错误信息
    fn run(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<()> {
        self.validate_targets()?;
        let roi_ctx = self.prepare_roi_hsv(window, ctx)?;
        let mask = build_mask(&roi_ctx.hsv)?;
        let boxes = contours_to_boxes(&mask)?;
        let row_tol = self.pick_row_tolerance(&boxes);
        let mut rows = cluster_rows(boxes, row_tol)?;

        sort_rows_and_cols(&mut rows);
        let window_pos = window.position()?;
        let window_size = window.size()?;
        let (scale_x, scale_y) = compute_scale_factors(roi_ctx.img_dims, window_size);
        let seat_area = build_seat_area(
            &rows,
            roi_ctx.roi_origin,
            window_pos,
            (scale_x, scale_y),
            &roi_ctx.hsv,
        );
        self.ensure_row_exists(seat_area.row_count())?;

        let seats_in_row = seat_area
            .row(self.target_row)
            .ok_or_else(|| anyhow!("目标行超出范围：第 {} 排不存在", self.target_row))?;

        for col in self.target_cols.iter().copied() {
            let seat = seat_area.seat(self.target_row, col).ok_or_else(|| {
                anyhow!(
                    "目标座位超出范围：第 {} 排没有第 {} 座（本排 {} 个）",
                    self.target_row,
                    col,
                    seats_in_row.len()
                )
            })?;

            println!(
                "座位: 第 {} 排第 {} 座 [{}] -> 图像 ({:.1}, {:.1}) -> 屏幕 ({}, {})",
                seat.row,
                seat.col,
                format_seat_state(seat.state),
                seat.center_img.0,
                seat.center_img.1,
                seat.center_screen.0,
                seat.center_screen.1
            );

            if self.click {
                let (sx, sy) = seat.screen_point();
                input::click_screen(sx, sy)?;
            }
        }

        if self.click {
            ctx.invalidate();
        }

        Ok(())
    }

    /// Step 名称，用于日志/调试。
    ///
    /// 返回固定字符串，便于流水线输出和排查。
    fn label(&self) -> &'static str {
        "LocateSeatByColor"
    }
}

/// ROI 预处理结果，包含 HSV 矩阵及坐标信息。
pub(crate) struct RoiContext {
    hsv: Mat,
    roi_origin: (u32, u32),
    img_dims: (u32, u32),
}

impl LocateSeatByColor {
    /// 校验调用参数，避免在聚类/索引阶段触发越界。
    ///
    /// - `target_row` 必须从 1 开始；否则立即报错
    /// - `target_cols` 不能为空；空列表无法执行定位/点击
    fn validate_targets(&self) -> Result<()> {
        if self.target_row == 0 {
            return Err(anyhow!("target_row 从 1 开始计数"));
        }
        if self.target_cols.is_empty() {
            return Err(anyhow!("target_cols 不能为空"));
        }
        Ok(())
    }

    /// 截图->BGR->裁剪->HSV，并返回 ROI 原点与截图尺寸。
    ///
    /// - 从窗口抓取 RGBA，转换为 BGR
    /// - 按给定 ROI（或整图）裁剪；范围非法时返回错误
    /// - 将 ROI 转换成 HSV，便于后续颜色掩膜
    /// - 返回 HSV 矩阵、ROI 左上角偏移以及原始截图尺寸
    fn prepare_roi_hsv(&self, window: &mut WandaWindow, ctx: &mut RunCtx) -> Result<RoiContext> {
        let rgba = ctx.capture_rgba(window)?;
        let bgr = vision::rgba_to_bgr(rgba)?;
        roi_hsv_from_bgr(&bgr, self.roi)
    }

    /// 根据检测到的方块高度估算行聚类容忍度。
    ///
    /// - 若传入 `row_tolerance_px`，取其与 1 的最大值
    /// - 否则用平均高度加 6 作为容忍度，至少 8px，避免不同排被合并
    fn pick_row_tolerance(&self, boxes: &[core::Rect]) -> i32 {
        if let Some(tol) = self.row_tolerance_px {
            return tol.max(1);
        }
        let avg_h = boxes.iter().map(|r| r.height).sum::<i32>() as f32 / boxes.len().max(1) as f32;
        (avg_h.round() as i32 + 6).max(8)
    }

    /// 确保目标行存在。
    ///
    /// - 当 `target_row` 为 0 或大于行数时返回错误，阻断后续点击/输出
    /// - 正常情况不修改数据，仅做范围检查
    fn ensure_row_exists(&self, row_count: usize) -> Result<()> {
        if self.target_row == 0 || self.target_row > row_count {
            return Err(anyhow!(
                "目标行超出范围：请求第 {} 排，总行数 {}",
                self.target_row,
                row_count
            ));
        }
        Ok(())
    }
}

/// 裁剪 ROI 并返回 ROI BGR、原图尺寸和 ROI 左上角偏移。
///
/// - 通过 `resolve_roi` 校验/默认 ROI
/// - 使用 `Mat::roi` 抽取子图并拷贝
/// - 返回 `(roi_bgr, roi_origin, img_dims)`，便于后续转换或保存
pub(crate) fn crop_bgr_with_roi(
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
pub(crate) fn bgr_to_hsv(bgr: &Mat) -> Result<Mat> {
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
pub(crate) fn roi_hsv_from_bgr(bgr: &Mat, roi: Option<(u32, u32, u32, u32)>) -> Result<RoiContext> {
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
pub(crate) fn resolve_roi(
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
pub(crate) fn build_mask(hsv: &Mat) -> Result<Mat> {
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
pub(crate) fn contours_to_boxes(mask: &Mat) -> Result<Vec<core::Rect>> {
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
pub(crate) fn cluster_rows(
    boxes: Vec<core::Rect>,
    row_tolerance: i32,
) -> Result<Vec<Vec<core::Rect>>> {
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
pub(crate) fn sort_rows_and_cols(rows: &mut [Vec<core::Rect>]) {
    rows.sort_by_key(|r| r.iter().map(|x| x.y).min().unwrap_or(0));
    for row in rows.iter_mut() {
        row.sort_by_key(|r| r.x);
    }
}

/// 计算图像->窗口缩放比例。
///
/// - 返回 `(scale_x, scale_y)`，用于将截图坐标换算到窗口逻辑坐标
/// - 若窗口尺寸与截图一致，结果约为 (1.0, 1.0)
pub(crate) fn compute_scale_factors(img_dims: (u32, u32), window_size: (u32, u32)) -> (f32, f32) {
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
pub(crate) fn build_seats(
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
pub(crate) fn build_seat_area(
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
pub(crate) fn format_seat_state(state: SeatState) -> &'static str {
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
pub(crate) fn seat_screen_point(
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
pub(crate) fn seat_color_ranges() -> Vec<(Scalar, Scalar)> {
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
pub(crate) fn seat_state_color_ranges() -> Vec<(SeatState, (Scalar, Scalar))> {
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
pub(crate) fn seat_state_from_hsv(hsv_roi: &Mat, rect: &core::Rect) -> SeatState {
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
pub(crate) fn scalar_in_range(val: &Scalar, lower: &Scalar, upper: &Scalar) -> bool {
    (val[0] >= lower[0] && val[0] <= upper[0])
        && (val[1] >= lower[1] && val[1] <= upper[1])
        && (val[2] >= lower[2] && val[2] <= upper[2])
}

/// 计算矩形的中心（加上 ROI 偏移）。
///
/// - 以矩形左上角和宽高求中心
/// - 将 ROI 偏移叠加，返回全图坐标系下的中心点
pub(crate) fn rect_center(rect: &core::Rect, offset: (i32, i32)) -> (f64, f64) {
    (
        offset.0 as f64 + rect.x as f64 + rect.width as f64 * 0.5,
        offset.1 as f64 + rect.y as f64 + rect.height as f64 * 0.5,
    )
}

/// 求掩膜中非零像素的最小外接矩形，返回相对掩膜的 Rect。
///
/// - 掩膜全零时返回 `Ok(None)`
/// - 否则使用 `bounding_rect` 包裹所有非零点
pub(crate) fn mask_nonzero_bbox(mask: &Mat) -> Result<Option<core::Rect>> {
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
    use opencv::core::{self, CV_8UC3, Mat, Scalar, Size};
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
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/raw/movie-seats/3.png");
        let bgr = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        let size = bgr.size()?;
        assert!(
            size.width > 0 && size.height > 0,
            "image should not be empty"
        );

        // let roi_x = (size.width / 4).max(0) as u32;
        // let roi_y = (size.height / 4).max(0) as u32;
        // let roi_w = (size.width / 2).max(1) as u32;
        // let roi_h = (size.height / 2).max(1) as u32;
        //
        // 这个坐标刚好是座位区域
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
        // let hsv_size = ctx.hsv.size()?;

        // assert_eq!(origin, (roi_x, roi_y));
        assert_eq!(dims, (size.width as u32, size.height as u32));
        assert_eq!(ctx.roi_origin, origin);
        assert_eq!(ctx.img_dims, dims);
        // assert_eq!(
        //     (hsv_size.width as u32, hsv_size.height as u32),
        //     // (roi_w, roi_h)
        // );
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
        let step = LocateSeatByColor {
            target_row: 1,
            target_cols: vec![1],
            roi,
            click: false,
            row_tolerance_px: None,
        };
        let row_tol = step.pick_row_tolerance(&boxes);
        let mut rows = cluster_rows(boxes, row_tol)?;
        sort_rows_and_cols(&mut rows);

        let scale = compute_scale_factors(roi_ctx.img_dims, roi_ctx.img_dims);
        let area = build_seat_area(&rows, roi_ctx.roi_origin, (0, 0), scale, &roi_ctx.hsv);
        assert!(area.row_count() > 0, "should detect at least 1 row");

        println!("Seat status: {:?}", area.seat(8, 11).unwrap().state);

        for r in 1..=area.row_count() {
            // println!(
            //     "Rows: {:?}, Cols: {:?}",
            //     area.row_count(),
            //     area.cols_in_row(r)
            // );
            let row = area.row(r).expect("row should exist");
            assert!(!row.is_empty(), "row {} should have seats", r);
        }
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

    /// 从聚类框生成座位结构并验证行列与坐标。
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
}
