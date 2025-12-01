use anyhow::{Result, anyhow};
use std::process::Command;
use std::{thread, time::Duration};
use xcap::{Window, image::RgbaImage};

/// 包装 `xcap::Window`，负责发现、激活与截图“万达”小程序窗口。
pub struct WandaWindow {
    window: Option<Window>,
}

impl WandaWindow {
    /// 创建未绑定窗口的实例。
    pub fn new() -> Self {
        Self { window: None }
    }

    /// 查找并缓存标题包含“万达”的窗口，后续操作都依赖此步骤。
    pub fn find_window(&mut self) -> Result<()> {
        if !self.window.is_none() {
            return Ok(());
        }

        let windows = Window::all()?;

        windows
            .iter()
            .for_each(|item| println!("window: {}", item.title().unwrap()));

        let wand_window = windows
            .iter()
            .find(|w| match w.title() {
                Ok(title) => title.contains("万达"),
                Err(_) => false,
            })
            .ok_or_else(|| anyhow!("未找到万达的小程序"))?;

        self.window = Some(wand_window.clone());
        Ok(())
    }

    /// 返回窗口的逻辑尺寸（宽、高）。
    pub fn size(&self) -> Result<(u32, u32)> {
        if let Some(window) = &self.window {
            Ok((window.width()?, window.height()?))
        } else {
            Err(anyhow!("未查找万达小程序"))
        }
    }

    /// 返回窗口左上角的屏幕坐标。
    pub fn position(&self) -> Result<(i32, i32)> {
        if let Some(window) = &self.window {
            Ok((window.x()?, window.y()?))
        } else {
            Err(anyhow!("未查找万达小程序"))
        }
    }

    /// 以 RGBA 格式捕获当前窗口内容。
    pub fn capture(&self) -> Result<RgbaImage> {
        if let Some(window) = &self.window {
            let image = window.capture_image()?;
            Ok(image)
        } else {
            Err(anyhow!("未查找万达小程序"))
        }
    }

    /// 将窗口置于最前方：优先按 PID 激活，失败时按应用名称激活。
    pub fn activate(&self) -> Result<()> {
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| anyhow!("未查找万达小程序"))?;
        let pid = window.pid()? as i32;
        let app_name = window.app_name()?;

        // Try focus by pid via System Events
        let script_pid = format!(
            "tell application \"System Events\" to set frontmost of (first process whose unix id is {pid}) to true"
        );
        let status_pid = Command::new("osascript").args(["-e", &script_pid]).status();

        let ok = matches!(status_pid, Ok(s) if s.success());
        if !ok {
            // Fallback: activate by app name
            let script_app = format!("tell application \"{app_name}\" to activate");
            let _ = Command::new("osascript").args(["-e", &script_app]).status();
        }

        // Give the OS a moment to bring it front
        thread::sleep(Duration::from_millis(150));
        Ok(())
    }
}
