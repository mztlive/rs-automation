use super::Step;
use crate::input;
use crate::pipeline::RunCtx;
use crate::window::WandaWindow;
use anyhow::Result;

/// 网格位置，按 3x3 九宫格定义。
#[derive(Clone, Copy)]
pub enum GridPos {
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    Center,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl GridPos {
    fn to_offsets(self) -> (f32, f32) {
        use GridPos::*;
        match self {
            TopLeft => (1.0 / 6.0, 1.0 / 6.0),
            TopCenter => (0.5, 1.0 / 6.0),
            TopRight => (5.0 / 6.0, 1.0 / 6.0),
            MiddleLeft => (1.0 / 6.0, 0.5),
            Center => (0.5, 0.5),
            MiddleRight => (5.0 / 6.0, 0.5),
            BottomLeft => (1.0 / 6.0, 5.0 / 6.0),
            BottomCenter => (0.5, 5.0 / 6.0),
            BottomRight => (5.0 / 6.0, 5.0 / 6.0),
        }
    }
}

/// 将鼠标移动到某个绝对坐标，或窗口内的九宫格位置。
pub struct MoveMouse {
    /// 直接指定屏幕坐标。如果提供则优先使用。
    pub screen_pos: Option<(i32, i32)>,
    /// 若未提供绝对坐标，则使用窗口内九宫格位置。
    pub grid_pos: Option<GridPos>,
}

impl MoveMouse {
    pub fn to_screen(x: i32, y: i32) -> Self {
        Self {
            screen_pos: Some((x, y)),
            grid_pos: None,
        }
    }

    pub fn to_grid(grid_pos: GridPos) -> Self {
        Self {
            screen_pos: None,
            grid_pos: Some(grid_pos),
        }
    }
}

impl Step for MoveMouse {
    fn run(&self, window: &mut WandaWindow, _ctx: &mut RunCtx) -> Result<()> {
        let (x, y) = if let Some((x, y)) = self.screen_pos {
            (x, y)
        } else {
            let grid = self
                .grid_pos
                .ok_or_else(|| anyhow::anyhow!("MoveMouse requires screen_pos or grid_pos"))?;
            let (win_x, win_y) = window.position()?;
            let (win_w, win_h) = window.size()?;
            let (ox, oy) = grid.to_offsets();
            let target_x = win_x + (win_w as f32 * ox).round() as i32;
            let target_y = win_y + (win_h as f32 * oy).round() as i32;
            (target_x, target_y)
        };

        input::move_mouse(x, y)
    }
}
