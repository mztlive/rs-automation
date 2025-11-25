use anyhow::{Context, Result, anyhow};
use std::ops::{Add, Sub};

/// 可在运行时计算的标量表达式，支持窗口尺寸变量与加/减。
#[derive(Clone, Debug)]
pub enum ScalarExpr {
    Literal(i32),
    WindowWidth,
    WindowHeight,
    Add(Box<ScalarExpr>, Box<ScalarExpr>),
    Sub(Box<ScalarExpr>, Box<ScalarExpr>),
}

/// 便捷访问窗口变量：`window.width - 50` 这样的语法会在运行时求值。
#[allow(non_upper_case_globals)]
pub const window: WindowExpr = WindowExpr {
    width: ScalarExpr::WindowWidth,
    height: ScalarExpr::WindowHeight,
};

/// 窗口维度占位符。
#[derive(Clone, Debug)]
pub struct WindowExpr {
    pub width: ScalarExpr,
    pub height: ScalarExpr,
}

impl ScalarExpr {
    /// 求值为 i64，避免中间结果溢出。
    pub fn eval_raw(&self, window_size: (u32, u32)) -> i64 {
        match self {
            ScalarExpr::Literal(v) => *v as i64,
            ScalarExpr::WindowWidth => window_size.0 as i64,
            ScalarExpr::WindowHeight => window_size.1 as i64,
            ScalarExpr::Add(lhs, rhs) => lhs.eval_raw(window_size) + rhs.eval_raw(window_size),
            ScalarExpr::Sub(lhs, rhs) => lhs.eval_raw(window_size) - rhs.eval_raw(window_size),
        }
    }

    /// 以 u32 形式求值，保证非负并在溢出时返回错误。
    pub fn eval_u32(&self, window_size: (u32, u32)) -> Result<u32> {
        let v = self.eval_raw(window_size);
        if v < 0 {
            Err(anyhow!("计算值为负数：{}", v))
        } else {
            u32::try_from(v).map_err(|_| anyhow!("计算值过大，超出 u32 范围：{}", v))
        }
    }
}

impl From<i32> for ScalarExpr {
    fn from(value: i32) -> Self {
        ScalarExpr::Literal(value)
    }
}

impl From<u32> for ScalarExpr {
    fn from(value: u32) -> Self {
        ScalarExpr::Literal(value as i32)
    }
}

impl<T> Add<T> for ScalarExpr
where
    T: Into<ScalarExpr>,
{
    type Output = ScalarExpr;

    fn add(self, rhs: T) -> Self::Output {
        ScalarExpr::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T> Sub<T> for ScalarExpr
where
    T: Into<ScalarExpr>,
{
    type Output = ScalarExpr;

    fn sub(self, rhs: T) -> Self::Output {
        ScalarExpr::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

/// 表示 ROI 等需要四个坐标的矩形表达式。
#[derive(Clone, Debug)]
pub struct RectExpr {
    pub x: ScalarExpr,
    pub y: ScalarExpr,
    pub width: ScalarExpr,
    pub height: ScalarExpr,
}

impl RectExpr {
    pub fn new<X, Y, W, H>(x: X, y: Y, width: W, height: H) -> Self
    where
        X: Into<ScalarExpr>,
        Y: Into<ScalarExpr>,
        W: Into<ScalarExpr>,
        H: Into<ScalarExpr>,
    {
        Self {
            x: x.into(),
            y: y.into(),
            width: width.into(),
            height: height.into(),
        }
    }

    /// 运行时求值，基于窗口尺寸将表达式展开为具体像素。
    pub fn eval(&self, window_size: (u32, u32)) -> Result<(u32, u32, u32, u32)> {
        Ok((
            self.x.eval_u32(window_size).context("计算 ROI.x 失败")?,
            self.y.eval_u32(window_size).context("计算 ROI.y 失败")?,
            self.width
                .eval_u32(window_size)
                .context("计算 ROI.width 失败")?,
            self.height
                .eval_u32(window_size)
                .context("计算 ROI.height 失败")?,
        ))
    }
}

impl From<(u32, u32, u32, u32)> for RectExpr {
    fn from(value: (u32, u32, u32, u32)) -> Self {
        RectExpr::new(value.0, value.1, value.2, value.3)
    }
}

impl From<(i32, i32, i32, i32)> for RectExpr {
    fn from(value: (i32, i32, i32, i32)) -> Self {
        RectExpr::new(value.0, value.1, value.2, value.3)
    }
}
