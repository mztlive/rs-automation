use anyhow::{Result, anyhow};
use std::process::Command;
use std::{thread, time::Duration};
use xcap::{Window, image::RgbaImage};

/// 窗口选择规则：按标题关键字与应用名称打分，自动挑选匹配度最高的窗口。
#[derive(Debug, Clone, Default)]
pub struct WindowSelector {
    pub title_keywords: Vec<String>,
    pub app_names: Vec<String>,
}

impl WindowSelector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_title_keywords<T>(mut self, keywords: T) -> Self
    where
        T: IntoIterator,
        T::Item: Into<String>,
    {
        self.title_keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_app_names<T>(mut self, names: T) -> Self
    where
        T: IntoIterator,
        T::Item: Into<String>,
    {
        self.app_names = names.into_iter().map(Into::into).collect();
        self
    }

    fn score(&self, window: &Window) -> Option<usize> {
        let title = window.title().unwrap_or_default();
        let app = window.app_name().unwrap_or_default();
        let title_hits = self
            .title_keywords
            .iter()
            .filter(|kw| title.contains(kw.as_str()))
            .count();

        let matches_title = title_hits > 0 || self.title_keywords.is_empty();
        let matches_app = self.app_names.is_empty()
            || self
                .app_names
                .iter()
                .any(|name| app.contains(name.as_str()));

        if !matches_title && !matches_app {
            return None;
        }

        let mut score = 0usize;
        if title_hits > 0 {
            score += title_hits * 4;
            if title_hits == self.title_keywords.len() && !self.title_keywords.is_empty() {
                score += 4;
            }
        }

        if matches_app && !self.app_names.is_empty() {
            score += 6;
        }

        if score == 0 { Some(1) } else { Some(score) }
    }

    fn describe(&self) -> String {
        let titles = if self.title_keywords.is_empty() {
            "任意标题".to_string()
        } else {
            self.title_keywords.join("|")
        };
        let apps = if self.app_names.is_empty() {
            "任意应用".to_string()
        } else {
            self.app_names.join("|")
        };
        format!("标题包含: {titles}; 应用匹配: {apps}")
    }
}

/// 包装 `xcap::Window`，负责发现、激活与截图目标窗口。
pub struct WandaWindow {
    window: Option<Window>,
}

impl WandaWindow {
    /// 创建未绑定窗口的实例。
    pub fn new() -> Self {
        Self { window: None }
    }

    /// 查找并缓存匹配给定规则的窗口，后续操作都依赖此步骤。
    pub fn find_window(&mut self, selector: &WindowSelector) -> Result<()> {
        if !self.window.is_none() {
            return Ok(());
        }

        let windows = Window::all()?;

        let mut scored = windows
            .into_iter()
            .filter_map(|w| selector.score(&w).map(|score| (score, w)))
            .collect::<Vec<_>>();

        if scored.is_empty() {
            return Err(anyhow!(
                "未找到满足条件的窗口，匹配规则：{}",
                selector.describe()
            ));
        }

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let (score, best) = scored.remove(0);
        println!(
            "选择窗口 [{}] (score={})",
            best.title().unwrap_or_else(|_| "<unknown>".to_string()),
            score
        );
        self.window = Some(best);
        Ok(())
    }

    /// 返回窗口的逻辑尺寸（宽、高）。
    pub fn size(&self) -> Result<(u32, u32)> {
        if let Some(window) = &self.window {
            Ok((window.width()?, window.height()?))
        } else {
            Err(anyhow!("未查找到目标窗口"))
        }
    }

    /// 返回窗口左上角的屏幕坐标。
    pub fn position(&self) -> Result<(i32, i32)> {
        if let Some(window) = &self.window {
            Ok((window.x()?, window.y()?))
        } else {
            Err(anyhow!("未查找到目标窗口"))
        }
    }

    /// 以 RGBA 格式捕获当前窗口内容。
    pub fn capture(&self) -> Result<RgbaImage> {
        if let Some(window) = &self.window {
            let image = window.capture_image()?;
            Ok(image)
        } else {
            Err(anyhow!("未查找到目标窗口"))
        }
    }

    /// 将窗口置于最前方：优先按 PID 激活，失败时按应用名称激活。
    pub fn activate(&self) -> Result<()> {
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| anyhow!("未查找到目标窗口"))?;
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
