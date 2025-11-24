use anyhow::Result;
use enigo::{Enigo, MouseButton, MouseControllable};
use std::{thread, time::Duration};

/// 将鼠标移动到指定屏幕坐标并模拟一次左键点击。
///
/// 为降低误触概率，会在移动后短暂停顿；若系统未授予辅助功能权限，返回错误。
pub fn click_screen(x: i32, y: i32) -> Result<()> {
    let mut enigo = Enigo::new();
    enigo.mouse_move_to(x, y);
    thread::sleep(Duration::from_millis(60));
    enigo.mouse_click(MouseButton::Left);
    Ok(())
}

/// 垂直滚动指定的刻度（正值向上，负值向下）。
///
/// 对 macOS，刻度单位通常较小，建议按需调整数量级。
pub fn scroll_vertical(lines: i32) -> Result<()> {
    let mut enigo = Enigo::new();
    enigo.mouse_scroll_y(lines);
    Ok(())
}

/// 水平滚动指定的刻度（正值向右，负值向左）。
pub fn scroll_horizontal(lines: i32) -> Result<()> {
    let mut enigo = Enigo::new();
    enigo.mouse_scroll_x(lines);
    Ok(())
}

/// 将鼠标移动到指定屏幕坐标，不点击。
pub fn move_mouse(x: i32, y: i32) -> Result<()> {
    let mut enigo = Enigo::new();
    enigo.mouse_move_to(x, y);
    Ok(())
}
