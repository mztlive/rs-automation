//! LocateSeatByColor 用于在电影选座页面上根据座位颜色定位目标排/列并可选点击。
//! 使用前需保证窗口已激活且座位区域完全可见（上方 legend、底部按钮可用 ROI 排除）。
//!
//! 调用方式（示例）：
//! ```
//! pipeline.push(LocateSeatByColor {
//!     target_row: 5,
//!     target_cols: vec![7, 8],
//!     roi: Some(RectExpr::new(0, 120, 1080, 620)),
//!     click: true,
//!     row_tolerance_px: None,
//! });
//! ```
//! 参数说明：
//! - `target_row`：要操作的行，1 为顶部第一排；越界会报错。
//! - `target_cols`：目标列列表，1 为最左；缺失或越界同样报错。
//! - `roi`：可选裁剪区域 `(x, y, w, h)`，默认全图；用于排除 legend/按钮以避免误识别，支持 `window.width - 50` 等计算。
//! - `click`：为 `true` 时执行屏幕点击，否则仅打印定位结果；点击后会调用 `ctx.invalidate()`。
//! - `row_tolerance_px`：行聚类的 y 容忍度，未设置时按检测到的方块高度自动估算。
//!
//! 内部流程概览：
//! 1. `ctx.capture_rgba` 获取窗口截图，`rgba_to_bgr` 转换并裁剪 ROI。
//! 2. 将 ROI 转 HSV，遍历 `seat_color_ranges` 构建掩膜，做闭运算去噪。
//! 3. `find_contours`+`bounding_rect` 得到方块；尺寸过滤后做行聚类。
//! 4. 行列排序后构建 `SeatArea`（二维座位网格），按行列索引目标座位。
//! 5. 打印状态与坐标，`click=true` 时执行点击并使上下文失效。
//! 常见调整项：修改 `roi` 以排除噪声，或改动颜色范围/尺寸阈值以适配不同主题与座位大小。
use anyhow::{Result, anyhow};
use opencv::core;

use crate::{
    input,
    pipeline::{RectExpr, RunCtx, Step},
    vision,
    window::WandaWindow,
};

use crate::seat::{
    RoiContext, SeatArea, build_mask, build_seat_area, cluster_rows, compute_scale_factors,
    contours_to_boxes, format_seat_state, roi_hsv_from_bgr, sort_rows_and_cols,
};

/// 通过颜色掩膜+轮廓聚类定位影厅座位的行/列，并输出或点击目标座位。
///
/// 适合颜色相对固定的座位图（如示例截图），默认识别灰色（可选）、绿色（已选）、红色（不可选）三类方块。
/// - `target_row`: 目标排，1 为最上方的第一排
/// - `target_cols`: 需操作的座位列号列表，1 为最左边
/// - `roi`: 可选的裁剪区域，避免把上方 legend 或下方按钮识别为座位；坐标基于截图像素，支持 `window.width - 50` 形式的计算
/// - `click`: 是否直接点击目标座位；否则只打印坐标
/// - `row_tolerance_px`: 行聚类的 y 方向容忍度，不填则用检测到的座位高度估计
pub struct LocateSeatByColor {
    pub target_row: usize,
    pub target_cols: Vec<usize>,
    pub roi: Option<RectExpr>,
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
        let window_size = window.size()?;
        let (roi_ctx, resolved_roi) = self.prepare_roi_hsv(window, ctx)?;
        println!("resolved_roi {:?}", resolved_roi);
        let mask = build_mask(&roi_ctx.hsv)?;
        let boxes = contours_to_boxes(&mask)?;
        let row_tol = self.pick_row_tolerance(&boxes);
        let mut rows = cluster_rows(boxes, row_tol)?;

        sort_rows_and_cols(&mut rows);
        let window_pos = window.position()?;
        let (scale_x, scale_y) = compute_scale_factors(roi_ctx.img_dims, window_size);
        let seat_area: SeatArea = build_seat_area(
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

        println!("Row 3, col {:?}", seat_area.cols_in_row(3));

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
    fn prepare_roi_hsv(
        &self,
        window: &mut WandaWindow,
        ctx: &mut RunCtx,
    ) -> Result<(RoiContext, Option<(u32, u32, u32, u32)>)> {
        let rgba = ctx.capture_rgba(window)?;
        let img_dims = rgba.dimensions();
        let roi = self.resolve_roi(img_dims)?;
        let bgr = vision::rgba_to_bgr(rgba)?;
        let roi_ctx = roi_hsv_from_bgr(&bgr, roi)?;
        Ok((roi_ctx, roi))
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

    /// 将 ROI 表达式按最新截图尺寸求值为具体坐标。
    fn resolve_roi(&self, img_dims: (u32, u32)) -> Result<Option<(u32, u32, u32, u32)>> {
        self.roi
            .as_ref()
            .map(|expr| expr.eval(img_dims))
            .transpose()
    }
}
