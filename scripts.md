rm -rf /root/DUET/result/visualizations/ETTh2.csv 2>/dev/null; cd /root/DUET 

python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh2.csv" \
    --strategy-args '{"horizon": 336, "save_vis": true, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 336, "k": 3, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 50, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": false}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "ETTh2/paper_fig7_test" 2>&1



python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh2.csv" \
    --strategy-args '{"horizon": 336, "save_vis": true, "num_rollings": 1}' \
    --model-name "duet.DUET" \
    --model-hyper-params '{"CI": 1, "batch_size": 64, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 336, "k": 3, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 50, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512, "use_quantum_block": true, "use_highway_gate": false}' \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "ETTh2/split_fig_test" 2>&1