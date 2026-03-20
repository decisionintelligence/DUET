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


class QuantumOTOCBlock(nn.Module):
    """
    Quantum-inspired feature mixing block - v8 极简版本

    设计理念：参考 Q-SSM (Quantum State Space Model) 的轻量设计

    关键改动：
    1. 使用简化的酉旋转门替代复杂的 Cayley 变换
    2. 轻量级特征混合 (无复杂 SE 门控)
    3. 作为辅助增强而非替换 Channel Transformer
    4. 残差优先 (alpha=0.9)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 极简设计：仅用少量参数
        # 酉旋转门参数 (RY-RX ansatz 风格)
        self.theta_real = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.1)
        self.theta_imag = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.1)

        # 轻量级门控
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, 1)
        )

        # 极简投影
        self.projection = nn.Linear(d_model, d_model)

        # 残差权重：让原始特征主导
        self.alpha = nn.Parameter(torch.tensor(0.9))

        self.norm = nn.LayerNorm(d_model)

    def _unitary_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的酉旋转门
        参考 Q-SSM 的 RY-RX ansatz 思想
        """
        B, N, H, D = x.shape

        # 归一化
        norm = x.abs().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_norm = x / norm

        # 将实数张量转换为复数形式
        x_complex = torch.complex(x_norm, torch.zeros_like(x_norm))

        # 计算旋转角度 (基于酉旋转)
        theta = torch.atan2(x_complex.abs(), 1.0 + x_complex.abs())
        phi = torch.angle(x_complex)

        # 应用酉旋转
        cos_t = torch.cos(theta + self.theta_real.unsqueeze(0).unsqueeze(0))
        sin_t = torch.sin(theta + self.theta_imag.unsqueeze(0).unsqueeze(0))

        # 酉旋转: U = cos(θ)I - i*sin(θ)*e^{iφ}
        real_part = cos_t * x_complex.real - sin_t * x_complex.imag
        imag_part = cos_t * x_complex.imag + sin_t * x_complex.real

        rotated = torch.complex(real_part, imag_part)

        return rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_residual = x

        # 轻量门控
        gate_weight = torch.sigmoid(self.gate(x))

        # 转换为多头格式
        x_reshaped = x.view(B, N, self.n_heads, self.d_k)

        # 应用酉旋转
        rotated = self._unitary_rotation(x_reshaped)

        # 恢复形状并转回实数
        rotated = rotated.view(B, N, D)
        rotated = torch.real(rotated)  # 复数转实数

        # 投影
        rotated = self.projection(rotated)

        # 门控混合
        z_out = gate_weight * rotated + (1 - gate_weight) * x

        z_out = self.norm(z_out)

        # 极保守残差连接
        z_out = self.alpha * z_out + (1 - self.alpha) * x_residual

        return z_out


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

        n_heads = getattr(config, "n_heads", 4)

        if self.use_quantum_block:
            self.quantum_block = QuantumOTOCBlock(config.d_model, n_heads=n_heads)

            if self.use_adaptive_fusion:
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

        if self.n_vars > 1:
            if self.quantum_block is not None:
                quantum_output = self.quantum_block(temporal_feature)

                if self.use_adaptive_fusion and self.adaptive_fusion is not None:
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

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        return output, L_importance
