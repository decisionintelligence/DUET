#!/bin/bash
# DUET Quantum v9 - 量子并行模块 - 超长预测 horizon 测试 (第二步)
# 测试 H=2000, 2160, 2880

cd /root/DUET

echo "=========================================="
echo "DUET Quantum v9 - 量子并行架构 - 超长预测测试"
echo "=========================================="

# H=2000
echo ""
echo ">>> 测试 H=2000..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 2000, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 2000, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_quantum_parallel": true}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 120000 \
    --save-path "ETTh1/DUET_quantum_v9_long_horizon_v2"

# H=2160
echo ""
echo ">>> 测试 H=2160..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 2160, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 2160, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_quantum_parallel": true}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 120000 \
    --save-path "ETTh1/DUET_quantum_v9_long_horizon_v2"

# H=2880
echo ""
echo ">>> 测试 H=2880..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 2880, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 2880, "k": 3, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_quantum_parallel": true}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 120000 \
    --save-path "ETTh1/DUET_quantum_v9_long_horizon_v2"

echo ""
echo "=========================================="
echo "量子 v9 模型测试完成!"
echo "=========================================="
