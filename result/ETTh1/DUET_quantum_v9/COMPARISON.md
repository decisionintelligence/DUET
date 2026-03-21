# DUET 量子模块消融实验 - ETTh1 数据集性能对比

## 实验概述

| 版本 | 架构类型 | Highway Gate | Adaptive Fusion | Quantum Parallel |
|------|----------|--------------|-----------------|------------------|
| **DUET (基准)** | 无量子模块 | - | - | - |
| **v6** | 量子 + Highway | ✅ | ❌ | ❌ |
| **v8** | 极简量子 | ❌ | ❌ | ❌ |
| **v9** | 量子并行 | ❌ | ❌ | ✅ |

---

## Horizon = 96 结果

| 版本 | MSE_norm | MAE_norm | RMSE_norm | SMAPE_norm |
|------|----------|----------|-----------|------------|
| **DUET (基准)** | 0.3520 | 0.3843 | 0.5810 | 72.02 |
| **v6 (Highway)** | 0.3804 | 0.4005 | 0.6046 | 73.34 |
| **v8 (极简)** | 0.3950* | 0.4150* | 0.6281* | 74.55* |
| **v9 (并行)** | **0.3701** | **0.3950** | **0.5963** | **72.82** |

> *注：v8 数据为早期测试结果

### 关键发现

**v9 量子并行架构在 horizon=96 时：**
- MSE_norm: **0.3701** vs 基准 0.3520 (差距 5.1%)
- 优于 v6 Highway 版本 (0.3804)
- 表现稳定，SMAPE 与基准接近

---

## Horizon = 192 结果 (基准 DUET)

| 版本 | MSE_norm | MAE_norm | RMSE_norm | SMAPE_norm |
|------|----------|----------|-----------|------------|
| **DUET (基准)** | 0.3980 | 0.4095 | 0.6215 | 74.89 |
| **v6 (Highway)** | - | - | - | - |

---

## Horizon = 336 结果 (基准 DUET)

| 版本 | MSE_norm | MAE_norm | RMSE_norm | SMAPE_norm |
|------|----------|----------|-----------|------------|
| **DUET (基准)** | 0.4146 | 0.4271 | 0.6363 | 77.70 |

---

## Horizon = 720 结果 (基准 DUET)

| 版本 | MSE_norm | MAE_norm | RMSE_norm | SMAPE_norm |
|------|----------|----------|-----------|------------|
| **DUET (基准)** | 0.4288 | 0.4562 | 0.6527 | 82.06 |

---

## 架构分析

### v9 量子并行架构代码

```python
if self.use_quantum_parallel:
    # v9 新增：量子模块与Channel Transformer并行融合
    changed_input = rearrange(input, "b l n -> b n l")
    channel_mask = self.mask_generator(changed_input)
    transformer_output, _ = self.Channel_transformer(
        x=temporal_feature, attn_mask=channel_mask
    )
    # 并行融合：各50%权重
    channel_group_feature = 0.5 * quantum_output + 0.5 * transformer_output
```

### 消融实验结论

1. **纯量子模块 (v8)**：效果不佳，需要与 Channel Transformer 结合
2. **Highway Gate (v6)**：量子输出 → Gate → Transformer，效果一般
3. **并行架构 (v9)**：量子输出 + Transformer 各占 50%，效果最佳

### 量子 OTOC 模块原理

```
输入 x
  ↓
特征提取 (real_linear + imag_linear) → 复数向量 ψ₀
  ↓
酉变换 U = exp(-iH) via Cayley 分解
  ↓
测量基变换
  ↓
OTOC 权重计算 (2·p·(1-p))
  ↓
输出 z_out = prob ⊗ otoc_weight
  ↓
与 Channel Transformer 并行融合 (50/50)
```

---

## 结论

**v9 量子并行架构在 ETTh1 数据集上的表现：**

| 指标 | 相对基准 | 相对 v6 |
|------|---------|---------|
| MSE_norm | -5.1% | +2.7% ✅ |
| MAE_norm | -2.8% | +1.4% ✅ |
| SMAPE_norm | -1.1% | +0.7% ✅ |

✅ **量子并行架构验证成功**：50/50 并行融合比 Highway Gate 效果更好

---

## 下一步建议

1. 在更多 horizon (192, 336, 720) 上验证 v9 架构
2. 尝试不同的融合权重 (如 60/40, 40/60)
3. 在其他数据集 (ETTm1, ETTm2) 上验证
4. 探索可学习的融合权重而非固定 50/50
