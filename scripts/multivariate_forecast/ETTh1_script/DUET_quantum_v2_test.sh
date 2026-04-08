#!/bin/bash
# DUET Quantum V2.2 - 量子损失函数消融实验
# 测试量子损失是否帮助模型持续优化
# V2.2：修复学习率衰减过快问题，使用 type3 调度

cd /root/DUET

echo "=========================================="
echo "DUET Quantum V2.2 - 量子损失消融实验"
echo "目标：验证量子损失是否能帮助模型持续优化"
echo "改进：使用 lradj=type3 温和学习率衰减"
echo "=========================================="

# H=96 (标准短期预测)
echo ""
echo ">>> 测试 H=96, seq_len=96, 量子损失权重=1.0..."
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --strategy-args '{"horizon": 96, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.15, "e_layers": 2, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 3, "loss": "MSE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 20, "seq_len": 96, "use_quantum_block": true, "use_quantum_parallel": true, "quantum_loss_weight": 1.0}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 600000 \
    --save-path "ETTh1/DUET_quantum_v22_test"

echo ""
echo "=========================================="
echo "DUET Quantum V2.2 测试完成!"
echo "=========================================="
