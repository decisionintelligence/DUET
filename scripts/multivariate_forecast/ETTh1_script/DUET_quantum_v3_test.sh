#!/bin/bash
# DUET Quantum V3 - 量子模块架构改进测试
# 核心改进：量子预处理 + 可学习跳跃连接

cd /root/DUET

echo "=========================================="
echo "DUET Quantum V3 - 量子预处理架构测试"
echo "=========================================="

# H=96 测试
echo ""
echo ">>> 测试 H=96, seq_len=96..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 96, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 3, "loss": "MSE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 20, "seq_len": 96, "use_quantum_block": true, "use_quantum_parallel": true, "quantum_loss_weight": 0.0}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 600000 \
    --save-path "ETTh1/DUET_quantum_v3_test"

# H=192 测试
echo ""
echo ">>> 测试 H=192, seq_len=192..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 192, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.1, "horizon": 192, "k": 3, "loss": "MSE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 20, "seq_len": 192, "use_quantum_block": true, "use_quantum_parallel": true, "quantum_loss_weight": 0.0}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 600000 \
    --save-path "ETTh1/DUET_quantum_v3_test"

echo ""
echo "=========================================="
echo "DUET Quantum V3 测试完成!"
echo "=========================================="
