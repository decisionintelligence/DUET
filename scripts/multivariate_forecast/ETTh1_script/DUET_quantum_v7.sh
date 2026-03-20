#!/bin/bash
# ETTh1 数据集 - DUET Quantum v7 (保守优化版本)
# 设计理念：保守改进，只做微小调整
# 1. 保留原始 OTOC 计算方式
# 2. 简化 SE 门控
# 3. 更保守的残差连接 (alpha=0.8)

# Horizon=96
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 96}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 512, "d_model": 512, "dropout": 0.2, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v7

# Horizon=192
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 192}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 512, "d_model": 512, "dropout": 0.2, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 192, "k": 2, "loss": "MAE", "lr": 0.0008, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 336, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v7

# Horizon=336
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 336}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 336, "k": 3, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v7

# Horizon=720
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 720}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 720, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v7
