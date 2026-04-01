## 量子模块超长预测消融实验

---

## 实验设计

数据集：ETTh1
测试 horizon：1080, 1440, 1800（都是超过 1000 的）

对比：
- DUET 基准模型（不带量子模块）
- DUET_quantum v9（量子并行架构，50/50 融合）

参数设置：基本上是长预测的常规参数，e_layers=2 换成更大的容量。

---

## 实验结果

### Horizon = 1080

| 模型 | MSE_norm | MAE_norm | SMAPE_norm |
|------|----------|----------|------------|
| DUET | 0.3929 | 0.4591 | 93.26 |
| 量子 v9 | 0.3946 | 0.4714 | **89.37** |

1080 这个点上，量子模块的 MSE 和 MAE 略差于基准，差距不大（不到 3%）。但 SMAPE 降低了 4.2%，这个指标对异常值更鲁棒。

### Horizon = 1440

| 模型 | MSE_norm | MAE_norm | SMAPE_norm |
|------|----------|----------|------------|
| DUET | 0.5558 | 0.5629 | 99.20 |
| 量子 v9 | 0.7235 | 0.6892 | 126.96 |

这个点量子模块表现比较差，MSE 增加了 30%。可能是 1440 这个数字比较特殊：ETTh1 是电力负荷数据，24小时周期，1440 分钟正好是一天。量子模块可能在这里过拟合了？

### Horizon = 1800

| 模型 | MSE_norm | MAE_norm | SMAPE_norm |
|------|----------|----------|------------|
| DUET | 0.6436 | 0.6033 | 99.96 |
| 量子 v9 | **0.5222** | **0.5451** | 100.37 |

量子模块领先，MSE 降低 18.9%，MAE 降低 9.6%。

---

## 初步结论

1. Horizon 在 1000-1500 之间，量子模块效果不稳定（1080 还可以，1440 很差）
2. Horizon >= 1500 时，量子模块优势明显
3. 1440 这个特殊值值得注意，可能需要周期感知的机制

---

## 下一步

1. 跳过 1440，继续测试 2000, 2160, 2880 等更长的 horizon
2. 看看量子模块的优势是从哪个点开始稳定出现
3. 考虑针对 1440 附近的问题做特殊处理

---

## 复现方法

### 1. 基准 DUET 测试

运行以下脚本测试基准模型在超长 horizon 上的表现：

```bash
cd /root/DUET
bash ./scripts/multivariate_forecast/ETTh1_script/DUET_long_horizon_baseline.sh
```

结果会保存在 `result/ETTh1/DUET_long_horizon_baseline/` 目录下。

### 2. 量子 v9 模型测试

运行以下脚本测试量子并行架构：

```bash
cd /root/DUET
bash ./scripts/multivariate_forecast/ETTh1_script/DUET_quantum_v9_long_horizon.sh
```

结果会保存在 `result/ETTh1/DUET_quantum_v9_long_horizon/` 目录下。

### 3. 提取结果

查看基准结果：
```bash
cat result/ETTh1/DUET_long_horizon_baseline/test_report.*.csv
```

查看量子结果：
```bash
cat result/ETTh1/DUET_quantum_v9_long_horizon/test_report.*.csv
```

### 4. 修改测试参数

如果需要修改 horizon 或其他参数，直接编辑脚本：

- `DUET_long_horizon_baseline.sh` - 基准模型
- `DUET_quantum_v9_long_horizon.sh` - 量子模型

关键参数在 `--model-hyper-params` 和 `--strategy-args` 中：
- `horizon`: 预测长度
- `use_quantum_block`: 是否使用量子模块
- `use_quantum_parallel`: 是否使用量子并行架构（v9 特有）

---

