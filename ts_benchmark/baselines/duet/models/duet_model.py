from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import (
    Mahalanobis_mask,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)


class GradientNormLayer(nn.Module):
    """
    V3 新增：梯度归一化层

    防止梯度消失/爆炸，稳定训练过程
    """

    def __init__(self, d_model: int, eps: float = 1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.scale * x_norm + self.shift


class AdaptiveQuantumGate(nn.Module):
    """
    V3 新增：自适应量子-经典融合门控

    核心思想：基于输入特征动态决定量子模块和经典模块的融合权重
    """

    def __init__(self, d_model: int):
        super().__init__()

        # 门控网络：根据量子输出和经典输出的差异动态调整权重
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # 初始偏置：让量子模块在训练初期有适度的贡献
        self.init_weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, quantum_feat: torch.Tensor, classical_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantum_feat: 量子模块输出 [B, N, D]
            classical_feat: 经典Transformer输出 [B, N, D]
        Returns:
            融合后的特征 [B, N, D]
        """
        # 合并两个特征
        concat_feat = torch.cat([quantum_feat, classical_feat], dim=-1)

        # 生成门控权重 [B, N, D]
        gate = self.gate_net(concat_feat)

        # 使用初始权重进行加权
        gate = gate * (1 - self.init_weight) + self.init_weight

        # 动态融合：gate 控制量子特征的贡献度
        # 当 gate 大时，更多使用量子特征
        # 当 gate 小时，更多使用经典特征
        fused = gate * quantum_feat + (1 - gate) * classical_feat

        return fused


class QuantumOTOCBlock(nn.Module):
    """
    Quantum-inspired feature mixing block - V3 自适应融合版

    V3 改进：
    1. 自适应跳跃连接：移除固定 alpha，使用输入依赖的门控
    2. 梯度归一化层：稳定训练
    3. 动态残差缩放：根据层深度调整残差强度

    论文参考：
    - Adaptive Skip Connections (Nick Ryan, 2024)
    - Multi-scale Feature Fusion with Adaptive Weight
    """

    def __init__(self, d_model: int, n_heads: int = 4, loss_weight: float = 1.0, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.loss_weight = loss_weight
        self.layer_idx = layer_idx

        # V3 新增：梯度归一化层
        self.grad_norm = GradientNormLayer(d_model)

        # V3 新增：自适应残差缩放
        # 浅层使用更强的残差（保持稳定性），深层使用更弱的残差（增加表达能力）
        init_residual_scale = 0.8 / (1 + layer_idx * 0.2)  # 初始层 ~0.8，深层 ~0.5
        self.residual_scale = nn.Parameter(torch.tensor(init_residual_scale))

        # 原始 Cayley 变换设计
        self.real_linear = nn.Linear(d_model, d_model)
        self.imag_linear = nn.Linear(d_model, d_model)

        self.se_fc1 = nn.Linear(d_model, d_model // 4)
        self.se_fc2 = nn.Linear(d_model // 4, d_model)

        nn_init = nn.init.eye_
        self.H_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        self.H_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        for i in range(n_heads):
            nn_init(self.H_real.data[i])
            nn.init.zeros_(self.H_imag.data[i])

        self.M_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        self.M_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        for i in range(n_heads):
            nn_init(self.M_real.data[i])
            nn_init(self.M_imag.data[i])

        self.projection = nn.Linear(d_model, d_model)

        # V3：移除固定的 alpha，使用动态残差缩放
        self.norm = nn.LayerNorm(d_model)

    def _cayley_unitary(self, H: torch.Tensor) -> torch.Tensor:
        d = H.size(-1)
        I = torch.eye(d, device=H.device, dtype=H.dtype)
        A = I + 0.5j * H
        B = I - 0.5j * H
        U = torch.linalg.solve(A, B)
        return U

    def _se_gate(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.mean(x, dim=1)
        s = F.relu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s))
        return s.unsqueeze(1)

    def _compute_quantum_loss(self) -> torch.Tensor:
        """
        V2.2 新增：增强的量子专项损失函数

        包含三个组件：
        1. 厄米特性损失：确保 H 和 M 是厄米矩阵
        2. 范数正则化：防止参数梯度消失或爆炸
        3. 正交性损失：鼓励参数矩阵具有多样性

        设计原理：量子模块参数需要更强的梯度信号才能持续优化
        """
        device = self.H_real.device
        loss = torch.tensor(0.0, device=device)

        # ===== 1. 厄米特性损失 =====
        # H 和 M 应该是厄米矩阵（自共轭）
        H_real_sym = (self.H_real + self.H_real.transpose(-2, -1)) / 2
        H_imag_skew = (self.H_imag - self.H_imag.transpose(-2, -1)) / 2
        loss_H = (H_real_sym - self.H_real).pow(2).mean() + \
                 (H_imag_skew - self.H_imag).pow(2).mean()

        M_real_sym = (self.M_real + self.M_real.transpose(-2, -1)) / 2
        M_imag_skew = (self.M_imag - self.M_imag.transpose(-2, -1)) / 2
        loss_M = (M_real_sym - self.M_real).pow(2).mean() + \
                 (M_imag_skew - self.M_imag).pow(2).mean()

        loss = loss + loss_H + loss_M

        # ===== 2. 范数正则化（V2.2 新增）=====
        # 鼓励参数有足够的激活，防止死亡梯度
        # 通过惩罚参数的 L2 范数过小或过大
        H_norm = self.H_real.pow(2).mean()
        M_norm = self.M_real.pow(2).mean()

        # 目标：H_norm 和 M_norm 应该在 [0.01, 1.0] 范围内
        target_norm = 0.1
        loss = loss + (H_norm - target_norm).pow(2) * 2.0
        loss = loss + (M_norm - target_norm).pow(2) * 2.0

        # ===== 3. 奇异值多样性损失（V2.2 新增）=====
        # 鼓励 H 矩阵具有非零奇异值，增加量子多样性
        for H in [self.H_real, self.M_real]:
            # 计算奇异值
            s = torch.linalg.svdvals(H.reshape(self.n_heads, -1))
            if s.sum() > 1e-6:
                # 计算奇异值的变异系数（标准差/均值）
                # 高变异系数意味着奇异值分布不均匀（多样性好）
                cv = s.std() / (s.mean() + 1e-6)
                # 添加一个小损失来鼓励适度的变异
                # 目标：cv 应该在 [0.5, 2.0] 范围内
                target_cv = 1.0
                loss = loss + (cv - target_cv).pow(2) * 0.5

        # 动态权重：默认值为 1.0
        return self.loss_weight * loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_residual = x

        se_weight = self._se_gate(x)

        real = self.real_linear(x)
        imag = self.imag_linear(x)
        psi_0 = torch.complex(real, imag)

        norm = psi_0.abs().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        psi_0 = psi_0 / norm
        psi_0 = psi_0.view(B, N, self.n_heads, self.d_k)

        H_real = (self.H_real + self.H_real.transpose(-2, -1)) / 2
        H_imag = (self.H_imag - self.H_imag.transpose(-2, -1)) / 2
        H = torch.complex(H_real, H_imag)

        M_real = (self.M_real + self.M_real.transpose(-2, -1)) / 2
        M_imag = (self.M_imag - self.M_imag.transpose(-2, -1)) / 2
        M_ham = torch.complex(M_real, M_imag)

        H = H / (H.abs().max().clamp_min(1e-3))
        M_ham = M_ham / (M_ham.abs().max().clamp_min(1e-3))

        U = self._cayley_unitary(H)
        M_basis = self._cayley_unitary(M_ham)

        psi_t = torch.einsum('bnhd,hde->bnhe', psi_0, U.conj().transpose(-2, -1))
        psi_measured = torch.einsum('bnhd,hde->bnhe', psi_t, M_basis)
        meas_prob = torch.abs(psi_measured) ** 2

        prob_matrix = torch.abs(U) ** 2
        otoc_matrix = 2.0 * (prob_matrix * (1.0 - prob_matrix))
        otoc_weight = F.softmax(otoc_matrix, dim=-1)

        z_out = torch.einsum('bnhd,hde->bnhe', meas_prob, otoc_weight.transpose(-2, -1))
        z_out = z_out.reshape(B, N, D)
        z_out = z_out * se_weight
        z_out = self.norm(z_out)
        z_out = self.projection(z_out)

        # V3 新增：使用可学习的残差缩放替代固定 alpha
        # residual_scale 控制量子特征相对于输入的贡献度
        z_out = self.residual_scale * z_out + (1 - self.residual_scale) * x_residual

        # V2 新增：计算量子损失
        quantum_loss = self._compute_quantum_loss()

        return z_out, quantum_loss


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块 - v6 优化版本

    优化点：
    1. 权重初始化改进
    2. 温度参数控制分布平滑度
    3. 更稳定的融合策略
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_k = d_model // n_heads

        # ===== 路径1：v2 风格的固定残差 =====
        self.v2_alpha = nn.Parameter(torch.tensor(0.5))

        # ===== 路径2：v3 风格的 Attention 融合 =====
        self.attn_q_proj = nn.Linear(d_model, d_model)
        self.attn_k_proj = nn.Linear(d_model, d_model)
        self.attn_v_proj = nn.Linear(d_model, d_model)
        self.attn_o_proj = nn.Linear(d_model, d_model)
        self.attn_alpha_query = nn.Parameter(torch.zeros(n_heads, d_k))

        # ===== 路径3：v4 风格的 Highway 门控 =====
        self.highway_fc1 = nn.Linear(d_model * 2, d_model)
        self.highway_fc2 = nn.Linear(d_model, d_model, bias=False)

        # ===== 自适应权重网络 =====
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # 使用Tanh替代ReLU，更稳定
            nn.Linear(d_model, 3)
        )

        # 优化5：温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 优化6：权重初始化为均匀
        nn.init.uniform_(self.fusion_net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.fusion_net[-1].bias)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, quantum_feat: torch.Tensor, transformer_feat: torch.Tensor) -> torch.Tensor:
        B, N, D = quantum_feat.shape
        d_k = D // self.n_heads

        # ===== 计算三种融合结果 =====

        # 方法1：固定残差 (v2风格)
        v2_out = self.v2_alpha * quantum_feat + (1 - self.v2_alpha) * transformer_feat

        # 方法2：Attention 融合 (v3风格)
        q = self.attn_q_proj(transformer_feat).view(B, N, self.n_heads, d_k)
        k = self.attn_k_proj(quantum_feat).view(B, N, self.n_heads, d_k)
        v = self.attn_v_proj(quantum_feat).view(B, N, self.n_heads, d_k)
        q = q + self.attn_alpha_query.unsqueeze(0).unsqueeze(1)
        scores = torch.einsum('bnhd,bnhd->bnh', q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        v3_out = torch.einsum('bnhd,bnh->bnhd', v, attn_weights).reshape(B, N, D)
        v3_out = self.attn_o_proj(v3_out) + transformer_feat

        # 方法3：Highway 门控 (v4风格)
        combined = torch.cat([quantum_feat, transformer_feat], dim=-1)
        gate = torch.sigmoid(self.highway_fc2(F.relu(self.highway_fc1(combined))))
        v4_out = gate * quantum_feat + (1 - gate) * transformer_feat

        # ===== 自适应权重融合 =====
        fusion_input = torch.mean(transformer_feat, dim=1)  # [B, D]
        fusion_logits = self.fusion_net(fusion_input)  # [B, 3]

        # 使用温度参数控制分布平滑度
        fusion_weights = F.softmax(fusion_logits / (self.temperature + 1e-6), dim=-1)  # [B, 3]

        # 融合三种结果
        w1 = fusion_weights[:, 0].view(B, 1, 1)
        w2 = fusion_weights[:, 1].view(B, 1, 1)
        w3 = fusion_weights[:, 2].view(B, 1, 1)

        output = w1 * v2_out + w2 * v3_out + w3 * v4_out
        output = self.norm(output)

        return output


class HighwayGate(nn.Module):
    """v4 风格的 Highway Gate"""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_fc1 = nn.Linear(d_model * 2, d_model)
        self.gate_fc2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(combined))))
        output = gate * x1 + (1 - gate) * x2
        return output


class AttentionResiduals(nn.Module):
    """v3 风格的 Attention Residuals"""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.alpha_query = nn.Parameter(torch.zeros(n_heads, d_model // n_heads))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, quantum_feat: torch.Tensor, transformer_feat: torch.Tensor) -> torch.Tensor:
        B, N, D = quantum_feat.shape
        d_k = D // self.n_heads

        q = self.q_proj(transformer_feat).view(B, N, self.n_heads, d_k)
        k = self.k_proj(quantum_feat).view(B, N, self.n_heads, d_k)
        v = self.v_proj(quantum_feat).view(B, N, self.n_heads, d_k)

        q = q + self.alpha_query.unsqueeze(0).unsqueeze(1)

        scores = torch.einsum('bnhd,bnhd->bnh', q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum('bnhd,bnh->bnhd', v, attn_weights).reshape(B, N, D)

        output = self.o_proj(attn_output) + transformer_feat
        output = self.norm(output)

        return output


class EnhancedPredictionHead(nn.Module):
    """
    优化3：增强的预测头

    替换简单的Linear层，使用：
    1. 瓶颈结构 (bottleneck)
    2. 多层感知机
    3. 残差连接
    """

    def __init__(self, d_model: int, pred_len: int, dropout: float = 0.1):
        super().__init__()

        # 瓶颈结构：d_model -> d_model/4 -> pred_len
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, pred_len)
        )

        # 直接映射作为残差
        self.skip = nn.Linear(d_model, pred_len)

        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 瓶颈输出
        bottleneck_out = self.bottleneck(x)

        # 残差连接
        skip_out = self.skip(x)

        # 融合
        out = self.residual_weight * bottleneck_out + (1 - self.residual_weight) * skip_out

        return out


class DUETModel(nn.Module):
    """
    DUET Model - v6 优化版本

    优化内容：
    1. QuantumOTOCBlock: 李雅普诺夫近似 + 特征调制
    2. EnhancedPredictionHead: 瓶颈结构预测头
    3. AdaptiveFusion: 温度参数 + 权重初始化
    """

    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)

        # Channel_transformer
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.use_quantum_block = getattr(config, "use_quantum_block", False)
        self.use_attention_residuals = getattr(config, "use_attention_residuals", False)
        self.use_highway_gate = getattr(config, "use_highway_gate", False)
        self.use_adaptive_fusion = getattr(config, "use_adaptive_fusion", False)
        self.use_enhanced_head = getattr(config, "use_enhanced_head", False)
        self.use_quantum_parallel = getattr(config, "use_quantum_parallel", False)  # v9新增
        # V3 新增：自适应量子-经典门控融合
        self.use_adaptive_quantum_gate = getattr(config, "use_adaptive_quantum_gate", True)
        # V2.1 新增：量子损失权重配置，默认值从 0.01 提高到 1.0
        self.quantum_loss_weight = getattr(config, "quantum_loss_weight", 1.0)

        n_heads = getattr(config, "n_heads", 4)

        if self.use_quantum_block:
            self.quantum_block = QuantumOTOCBlock(
                config.d_model, n_heads=n_heads, loss_weight=self.quantum_loss_weight, layer_idx=0
            )

            # V3 新增：自适应门控融合模块
            if self.use_adaptive_quantum_gate and self.use_quantum_parallel:
                self.quantum_gate = AdaptiveQuantumGate(config.d_model)
            else:
                self.quantum_gate = None

            if self.use_quantum_parallel:
                # v9新增：并行架构
                self.adaptive_fusion = None
                self.attn_residuals = None
                self.highway_gate = None
            elif self.use_adaptive_fusion:
                self.adaptive_fusion = AdaptiveFusion(config.d_model, n_heads=n_heads)
                self.attn_residuals = None
                self.highway_gate = None
            elif self.use_highway_gate:
                self.highway_gate = HighwayGate(config.d_model)
                self.attn_residuals = None
                self.adaptive_fusion = None
            elif self.use_attention_residuals:
                self.attn_residuals = AttentionResiduals(config.d_model, n_heads=n_heads)
                self.highway_gate = None
                self.adaptive_fusion = None
            else:
                self.attn_residuals = None
                self.highway_gate = None
                self.adaptive_fusion = None
        else:
            self.quantum_block = None
            self.attn_residuals = None
            self.highway_gate = None
            self.adaptive_fusion = None

        # V3.2 新增：可学习的跳跃连接权重
        self.quantum_skip_weight = nn.Parameter(torch.tensor([0.3]))

        # 优化3：使用增强的预测头
        if self.use_enhanced_head:
            self.linear_head = EnhancedPredictionHead(
                config.d_model, config.pred_len, config.fc_dropout
            )
        else:
            self.linear_head = nn.Sequential(
                nn.Linear(config.d_model, config.pred_len),
                nn.Dropout(config.fc_dropout),
            )

    def forward(self, input):
        if self.CI:
            channel_independent_input = rearrange(input, "b l n -> (b n) l 1")
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=input.shape[0]
            )
        else:
            temporal_feature, L_importance = self.cluster(input)

        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")

        # V2 新增：初始化量子损失
        quantum_loss = None

        if self.n_vars > 1:
            if self.quantum_block is not None:
                quantum_output, quantum_loss = self.quantum_block(temporal_feature)

                if self.use_quantum_parallel:
                    # V3.1 新思路：量子模块作为预处理
                    # 先用量子模块预处理 temporal_feature
                    quantum_preprocessed = quantum_output

                    # 然后将预处理后的特征送入 Channel Transformer
                    changed_input = rearrange(input, "b l n -> b n l")
                    channel_mask = self.mask_generator(changed_input)
                    channel_group_feature, _ = self.Channel_transformer(
                        x=quantum_preprocessed, attn_mask=channel_mask
                    )

                    # V3.2 新增：添加跳跃连接，让模型可以学习是否使用量子预处理
                    # 使用可学习的融合权重
                    channel_group_feature = (
                        self.quantum_skip_weight * channel_group_feature +
                        (1 - self.quantum_skip_weight) * quantum_preprocessed
                    )

                elif self.use_adaptive_fusion and self.adaptive_fusion is not None:
                    changed_input = rearrange(input, "b l n -> b n l")
                    channel_mask = self.mask_generator(changed_input)
                    transformer_output, _ = self.Channel_transformer(
                        x=temporal_feature, attn_mask=channel_mask
                    )
                    channel_group_feature = self.adaptive_fusion(quantum_output, transformer_output)

                elif self.use_highway_gate and self.highway_gate is not None:
                    changed_input = rearrange(input, "b l n -> b n l")
                    channel_mask = self.mask_generator(changed_input)
                    transformer_output, _ = self.Channel_transformer(
                        x=temporal_feature, attn_mask=channel_mask
                    )
                    channel_group_feature = self.highway_gate(quantum_output, transformer_output)

                elif self.use_attention_residuals and self.attn_residuals is not None:
                    changed_input = rearrange(input, "b l n -> b n l")
                    channel_mask = self.mask_generator(changed_input)
                    transformer_output, _ = self.Channel_transformer(
                        x=temporal_feature, attn_mask=channel_mask
                    )
                    channel_group_feature = self.attn_residuals(quantum_output, transformer_output)

                else:
                    channel_group_feature = quantum_output
            else:
                changed_input = rearrange(input, "b l n -> b n l")
                channel_mask = self.mask_generator(changed_input)
                channel_group_feature, attention = self.Channel_transformer(
                    x=temporal_feature, attn_mask=channel_mask
                )
                quantum_loss = L_importance  # 非量子模式下用 L_importance

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)
            quantum_loss = L_importance  # n_vars=1 时也用 L_importance

        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        # V2 新增：返回量子损失以便在训练时整合
        total_quantum_loss = quantum_loss
        return output, total_quantum_loss
