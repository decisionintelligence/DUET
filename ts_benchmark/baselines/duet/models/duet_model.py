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
    Quantum-inspired feature mixing block - v4/v5 版本
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.real_linear = nn.Linear(d_model, d_model)
        self.imag_linear = nn.Linear(d_model, d_model)

        # SE 门控
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
        self.alpha = nn.Parameter(torch.tensor(0.5))
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
        z_out = self.alpha * z_out + (1 - self.alpha) * x_residual

        return z_out


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块 - v5 核心创新

    结合 v2 (固定残差), v3 (Attention), v4 (Highway) 三种融合方式的优点：
    1. 可学习的融合权重
    2. 特征调制
    3. 多尺度特征提取
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
        # 学习当前输入适合用哪种融合方式
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)  # 3种融合方式的权重
        )

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
        # 用 transformer 特征来决定权重
        fusion_input = torch.mean(transformer_feat, dim=1)  # [B, D]
        fusion_weights = F.softmax(self.fusion_net(fusion_input), dim=-1)  # [B, 3]

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


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)

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
