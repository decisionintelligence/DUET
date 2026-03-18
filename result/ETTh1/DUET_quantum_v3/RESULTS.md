# DUET Quantum V3 (Attention Residuals) 实验结果

## ETTh1 数据集

| Horizon | MSE_norm | MAE_norm | RMSE_norm |
|---------|----------|----------|-----------|
|      96 |   0.3738 |   0.4012 |    0.5996 |
|      192 |   0.4081 |   0.4166 |    0.6289 |
|      336 |   0.4152 |   0.4301 |    0.6372 |
|      720 |   0.4600 |   0.4772 |    0.6756 |

## 改进点

- 多 head 细粒度混合
- 残差连接
- 改进的 Hamiltonian 初始化
- Attention Residuals 动态融合
