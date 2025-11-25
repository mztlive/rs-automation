#![recursion_limit = "256"]

mod actions;
mod input;
mod page;
mod pipeline;
mod seat;
mod vision;
mod window;

use anyhow::Result;
use std::env;
use std::time::Instant;

/// CLI 入口：等待用户输入 `a`，随后激活目标窗口并运行页面流水线。
fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let movie_name = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("缺少参数：影片名称"))?;
    let show_date = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("缺少参数：观影日期"))?;
    let cinema_name = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("缺少参数：影院名称"))?;
    let show_time = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("缺少参数：场次时间"))?;

    println!(
        "将查找影片 [{}]，日期 [{}]，影院 [{}]，场次 [{}]",
        movie_name, show_date, cinema_name, show_time
    );

    let start = Instant::now();
    page::init("assets/page.json")?;
    let mut window = window::WandaWindow::new();
    window.find_window()?;

    window.activate()?;

    let booking = pipeline::BookingRequest::new(movie_name, show_date, cinema_name, show_time);
    let mut ctx = pipeline::RunCtx::with_booking_request(booking.clone());

    let pipeline = pipeline::default_pipeline(&booking);
    if let Err(e) = pipeline.run(&mut window, &mut ctx) {
        anyhow::bail!("执行页面管线失败: {e:?}");
    }

    println!("总耗时: {:?}", start.elapsed());

    Ok(())
}
