# Repository Guidelines

## 项目结构与模块组织
- `automation`：CLI 入口位于 `automation/src/main.rs`，负责窗口发现、激活与屏幕抓取；`window.rs`、`vision.rs`、`actions.rs`、`input.rs`、`page.rs` 分别处理窗口控制、OpenCV 工具、动作组合、输入驱动与页面配置解析。流水线实现集中在 `automation/src/pipeline/`，并以 `steps/` 拆分激活、等待、点击等步骤，便于复用和扩展。
- `ml`：基于 Burn 的轻量级识图模型，提供模板分类与特征提取能力。`automation` 在识图阶段通过该 crate 进行页面判断，`assets/page.json` 中的模板需与模型保持同步。
- `image-proc`：针对 OpenCV 的几何/颜色处理封装，提供 `translate`、`scale_preserve`、`rotate`、`adjust_color` 等细粒度 API，用于生成训练数据或后续图像流程。
- `image-pre-cli`：依赖 `image-proc` 的命令行工具，用于批量截图增强、写出 manifest 以及复现随机扰动（支持 `--seed`）。适合在整理 `dataset/` 时生产新的训练样本。
- ⚠️ `api`：**仅用于早期协议/wasm 试验，不参与当前自动化流程，也不是 workspace 成员。请勿依赖或发布该 crate 的任何代码。**
- 其他目录：`assets/` 存放页面描述及模板图；`dataset/` 用于原始/增强图片集；`artifacts/` 保存模型或抓取输出；`target/` 为构建产物并已忽略。

## 构建、测试与开发命令
- `cargo build`：一次性编译 `automation`、`ml`、`image-proc` 与 `image-pre-cli`，用于确认依赖齐备；`api` 默认不会被 workspace 构建。
- `cargo test`：运行所有单元测试；与 GUI/图像交互的测试默认标记为 `#[ignore]`，需要时可执行 `cargo test -- --ignored`。
- `cargo run -p automation`：按照提示（输入 `a`）完成窗口激活、页面识别与预置动作。
- `cargo run -p ml --example inspector`：调试 Burn 模型输出或验证模板分类效果。
- `cargo run -p image-pre-cli -- --help`：查看数据增强参数；常见用法为指定 `--input`、`--output`、`--variants` 与 `--manifest` 生成训练清单。

## 编码风格与命名约定
全部 Rust 代码遵循 `rustfmt` 默认风格。模块文件使用 `snake_case`，类型采用 `UpperCamelCase`，函数与变量保持 `snake_case`。公共 API 面向组合式小函数，便于管线中的步骤复用。错误处理统一以 `anyhow::Result` 返回，除测试外避免 `unwrap()`。

## 测试与资源指南
在实现所在文件中定义 `#[cfg(test)] mod tests`，测试函数命名 `test_*` 并保持确定性。当测试依赖窗口、OpenCV 或 GPU 输出时，请添加 `#[ignore]` 并在注释中解释启用条件。需要图片夹具时放入 `assets/` 或 `dataset/` 并记录来源，避免使用本地绝对路径。

## 提交与拉取请求规范
提交信息建议遵循 `feat:`、`fix:`、`refactor:`、`docs:` 等前缀，并聚焦单一变更。拉取请求应描述变更、列出验证步骤、关联 issue，并在涉及识图或 UI 变更时附日志/截图，方便复现。

## 安全与运行提示
勿提交密钥或个人路径。运行 `automation` 前需为 Terminal 授予 macOS 辅助功能权限，并确认本地已安装 OpenCV 与 `libclang`。在调用模型前请校验 `assets/page.json` 与当前窗口布局一致，以维持识别准确度；任何对 `api` 的试验改动都应保持与主流水线隔离。
