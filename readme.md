# 预处理
cargo run -p image-pre-cli -- --input=assets/raw --output=dataset --manifest=manifest.csv --label_source=parent-dir

# 训练
cargo run --bin train -- \
    --manifest=manifest.csv \
    --image-root=dataset \
    --output-dir=artifacts/page-classifier \
    --epochs=2 \
    --batch-size=4

# 推理
cargo run --bin infer -- --model-dir=artifacts/page-classifier --image=assets/raw/movie-list/list1.png


# pipeline
cargo run -p wanda-automation
