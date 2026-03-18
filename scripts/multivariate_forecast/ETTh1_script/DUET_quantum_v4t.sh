#!/bin/bash
# DUET Quantum V4 + 调优版本
# 关键改进：调整学习率和 dropout

# H=96: 降低dropout，提高学习率
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 96}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 512, "d_model": 512, "dropout": 0.3, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v4t

# H=192: 平衡设置
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 192}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 512, "d_model": 512, "dropout": 0.3, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 192, "k": 2, "loss": "MAE", "lr": 0.0008, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 336, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v4t

# H=336: 提高学习率
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 336}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.3, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 336, "k": 3, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v4t

# H=720: 长预测，适当降低dropout
python ./scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list ETTh1.csv --strategy-args '{"horizon": 720}' --model-name duet.DUET --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.05, "horizon": 720, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": true}' --deterministic full --gpus 0 --num-workers 1 --timeout 60000 --save-path ETTh1/DUET_quantum_v4t
