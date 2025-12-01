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
    let args: Vec<String> = env::args().skip(1).collect();
    let (pipeline_name, offset) = match args.len() {
        4 => ("wanda".to_string(), 0),
        5 => (args[0].clone(), 1),
        _ => {
            anyhow::bail!(
                "参数数量不正确。用法：<pipeline 可选> <影片名称> <观影日期> <影院名称> <场次时间>"
            )
        }
    };
    let movie_name = args[offset].clone();
    let show_date = args[offset + 1].clone();
    let cinema_name = args[offset + 2].clone();
    let show_time = args[offset + 3].clone();

    println!(
        "使用 pipeline [{}]，将查找影片 [{}]，日期 [{}]，影院 [{}]，场次 [{}]",
        pipeline_name, movie_name, show_date, cinema_name, show_time
    );

    let start = Instant::now();
    let booking = pipeline::BookingRequest::new(movie_name, show_date, cinema_name, show_time);
    let pipeline = pipeline::resolve_pipeline(&pipeline_name, &booking)?;
    println!("加载页面配置 [{}]", pipeline.page_config_path);
    let mut ctx = pipeline::RunCtx::with_booking_request(booking.clone());

    if let Err(e) = pipeline.run(&mut ctx) {
        anyhow::bail!("执行页面管线失败: {e:?}");
    }

    println!("总耗时: {:?}", start.elapsed());

    Ok(())
}
