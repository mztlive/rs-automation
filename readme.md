# 视图自动化（中文指南）

Rust workspace，用于桌面视图自动化：窗口发现/激活、OCR+模板识别、流水线点击订票，包含 Burn 模型和数据处理工具。

## 目录结构
- `automation/`：CLI 与流水线；窗口控制、OCR/模板步骤与可组合的 Step。
- `ml/`：Burn 模型（页面/模板分类）。
- `image-proc/`：OpenCV 几何/颜色处理封装。
- `image-pre-cli/`：数据增强与 manifest 生成 CLI。
- `assets/`：页面模板与配置；`dataset/`：训练数据；`artifacts/`：模型与输出。

## 前置条件
- macOS，Terminal 已授予辅助功能权限。
- 本地安装 OpenCV 与 `libclang`。
- Rust toolchain（遵循 `rustfmt` 默认风格）。

## 构建与测试
- 构建全部 crate：`cargo build`
- 运行测试：`cargo test`（GUI/图像相关测试默认 `#[ignore]`，需要时 `cargo test -- --ignored`）

## 自动化 CLI（非交互）
按顺序传四个参数：`影片名称` `观影日期` `影院名称` `场次时间`。
```
cargo run -p wanda-automation -- "窗外是蓝星" "11月24日" "影院示例" "09:20"
```
行为概览：
- 定位并激活目标窗口。
- 进入电影页，OCR 滚动找到影片并点击。
- OCR 选择日期（含横向滚动）、选择影院，再选择场次时间。
- 文本匹配使用 NFKC 归一化 + 去空白 + 包含判断 + Levenshtein/LCS 模糊匹配，提升中文/OCR 鲁棒性。

## 数据准备与训练
- 数据增强与 manifest：
  ```
  cargo run -p image-pre-cli -- --input=assets/raw --output=dataset --manifest=manifest.csv --label_source=parent-dir
  ```
- 训练（Burn）：
  ```
  cargo run --bin train -- \
    --manifest=manifest.csv \
    --image-root=dataset \
    --output-dir=artifacts/page-classifier \
    --epochs=2 \
    --batch-size=4
  ```
- 推理验证：
  ```
  cargo run --bin infer -- --model-dir=artifacts/page-classifier --image=assets/raw/movie-list/list1.png
  ```

## 调试与工具
- 模型输出检查：`cargo run -p ml --example inspector`
- 流水线步骤：`automation/src/pipeline/steps/`
- 日志：stdout 输出，带步骤标签，便于定位。

## 注意事项
- 保持 `assets/page.json` 与模板/模型一致。
- `api` crate 为实验，不在当前流水线内，请勿依赖。
