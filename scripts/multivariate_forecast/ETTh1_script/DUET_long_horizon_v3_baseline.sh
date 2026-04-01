#!/bin/bash
# DUET 基准模型 - 第三步测试
# 策略：增大 seq_len=1024，测试 H=2000, 2160, 3600, 4320

cd /root/DUET

echo "=========================================="
echo "DUET 基准模型 - seq_len=1024 测试"
echo "=========================================="

# H=2000 (seq_len=1024)
echo ""
echo ">>> 测试 H=2000, seq_len=1024..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 2000, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 2000, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 1024}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 180000 \
    --save-path "ETTh1/DUET_long_horizon_v3_baseline"

# H=2160 (seq_len=1024)
echo ""
echo ">>> 测试 H=2160, seq_len=1024..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 2160, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 2160, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 1024}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 180000 \
    --save-path "ETTh1/DUET_long_horizon_v3_baseline"

# H=3600 (seq_len=1024)
echo ""
echo ">>> 测试 H=3600, seq_len=1024..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 3600, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 3600, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 1024}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 180000 \
    --save-path "ETTh1/DUET_long_horizon_v3_baseline"

# H=4320 (seq_len=1024)
echo ""
echo ">>> 测试 H=4320, seq_len=1024..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 4320, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 4320, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 1024}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 180000 \
    --save-path "ETTh1/DUET_long_horizon_v3_baseline"

echo ""
echo "=========================================="
echo "基准 DUET 测试完成!"
echo "=========================================="
