use super::steps::*;
use crate::pipeline::{BookingRequest, Pipeline, PipelineBundle, RectExpr, window};
use crate::window::WindowSelector;
use std::time::Duration;

/// Wanda 小程序流水线：绑定自身的窗口选择规则与页面配置。
pub fn wanda_pipeline(request: &BookingRequest) -> PipelineBundle {
    PipelineBundle::new(
        "wanda",
        "assets/page.json",
        WindowSelector::new()
            .with_title_keywords(["万达"])
            .with_app_names(["WeChat", "微信"]),
        wanda_steps(request),
    )
}

/// 统一入口流水线：内部会尝试回退到一级页面后执行正常流程。
fn wanda_steps(request: &BookingRequest) -> Pipeline {
    Pipeline::new()
        .step(LoopUntil {
            label: "ensure-first-level",
            cond: Condition::AnchorAbove {
                page: "common",
                anchor: "bottom_nav_bar",
            },
            on_miss: Sequence::new().step(ClickWindowPos {
                pos: WindowPos::Logical { x: 20, y: 50 }, // 未来改成读取配置
            }),
            max_iters: 5,
            delay_ms: Some(1000),
        })
        .step(SleepMs(1500))
        .step(RetryStep::new(
            "click-nav-movie-primary",
            ClickAnchor {
                page: "common",
                anchor: "bottom_nav_movie",
                pos: AnchorClickPos::Center,
            },
            3,
            Duration::from_millis(120),
        ))
        .step(SleepMs(150))
        .step(WaitPage {
            targets: &["movie-list"],
            min_score: 0.6,
            timeout_ms: 5000,
            poll_ms: 150,
        })
        .step(DebugStep::new("进入了电影页面"))
        .step(MoveMouse {
            screen_pos: None,
            grid_pos: Some(GridPos::Center),
        })
        .step(ScrollThenOcrLoop {
            patterns: request.movie_name.clone(),
            max_attempts: 10,
        })
        .step(DebugStep::new("找到了电影"))
        .step(ClickOcrMatch {
            patterns: request.movie_name.clone(),
            case_sensitive: false,
        })
        .step(DebugStep::new("点击了电影"))
        .step(SleepMs(3000))
        .step(DebugStep::new("等待页面加载"))
        .step(ClickAnchor {
            page: "common",
            anchor: "quick_buy_movie",
            pos: AnchorClickPos::Center,
        })
        .step(SleepMs(3000))
        .step(SelectDateByOcr {
            target_pattern: request.show_date.clone(),
            max_scroll_attempts: 8,
            scroll_pixels: 10,
        })
        .step(DebugStep::new("选择了日期"))
        .step(SleepMs(3000))
        .step(MoveMouse::to_grid(GridPos::Center))
        .step(ScrollThenOcrLoop {
            patterns: request.cinema_name.clone(),
            max_attempts: 10,
        })
        .step(ClickOcrMatch {
            patterns: request.cinema_name.clone(),
            case_sensitive: false,
        })
        .step(DebugStep::new("点击了影院"))
        .step(SleepMs(3000))
        .step(MoveMouse::to_grid(GridPos::BottomCenter))
        .step(ScrollThenOcrLoop {
            patterns: request.show_time.clone(),
            max_attempts: 6,
        })
        .step(ClickOcrMatch {
            patterns: request.show_time.clone(),
            case_sensitive: false,
        })
        .step(DebugStep::new("选择了场次时间"))
        .step(SleepMs(3000))
        .step(ConditionalStep {
            label: "是否有确认弹窗",
            cond: Condition::AnchorAbove {
                page: "common",
                anchor: "confirm-btn",
            },
            then_seq: ClickAnchor {
                page: "common",
                anchor: "confirm-btn",
                pos: AnchorClickPos::Center,
            },
            else_seq: None,
        })
        .step(SleepMs(3000))
        .step(LocateSeatByColor {
            target_row: 1,
            target_cols: vec![1, 2],
            // 示例裁剪范围，按实际截图可调整
            roi: Some(RectExpr::new(50, 700, window.width - 50, 530)),
            click: true,
            row_tolerance_px: None,
        })
        .step(SleepMs(3000))
}
