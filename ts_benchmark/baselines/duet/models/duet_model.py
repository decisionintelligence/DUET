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
    Quantum-inspired feature mixing block - v4 版本

    改进点：
    1. 多 head 细粒度混合
    2. SE (Squeeze-and-Excitation) 门控机制
    3. 改进的 Hamiltonian 初始化
    4. 可学习残差权重
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 将实值特征映射到复数态
        self.real_linear = nn.Linear(d_model, d_model)
        self.imag_linear = nn.Linear(d_model, d_model)

        # SE 门控机制
        self.se_fc1 = nn.Linear(d_model, d_model // 4)
        self.se_fc2 = nn.Linear(d_model // 4, d_model)

        # 多 head 的哈密顿量 H 和测量算子 M
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

        # 投影层
        self.projection = nn.Linear(d_model, d_model)

        # 可学习残差权重
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
        """SE 门控机制"""
        s = torch.mean(x, dim=1)
        s = F.relu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s))
        return s.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_residual = x

        # SE 门控
        se_weight = self._se_gate(x)

        # 量子态构造
        real = self.real_linear(x)
        imag = self.imag_linear(x)
        psi_0 = torch.complex(real, imag)

        norm = psi_0.abs().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        psi_0 = psi_0 / norm

        # 多 head 变换
        psi_0 = psi_0.view(B, N, self.n_heads, self.d_k)

        # 量子演化
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

        # 应用 SE 门控
        z_out = z_out * se_weight

        z_out = self.norm(z_out)
        z_out = self.projection(z_out)

        # 残差连接
        z_out = self.alpha * z_out + (1 - self.alpha) * x_residual

        return z_out


class HighwayGate(nn.Module):
    """
    Highway Network 风格的门控机制

    结合固定残差和动态 Attention 的优点：
    - 使用简单的门控网络，学习何时使用量子特征
    - 比 Attention 更轻量
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 门控网络
        self.gate_fc1 = nn.Linear(d_model * 2, d_model)
        self.gate_fc2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1: 量子特征, x2: Transformer 特征"""
        combined = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(combined))))
        output = gate * x1 + (1 - gate) * x2
        return output


class AttentionResiduals(nn.Module):
    """Attention Residuals 模块 - v3 版本"""

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

        q = self.q_proj(transformer_feat)
        k = self.k_proj(quantum_feat)
        v = self.v_proj(quantum_feat)

        d_k = D // self.n_heads
        q = q.view(B, N, self.n_heads, d_k)
        k = k.view(B, N, self.n_heads, d_k)
        v = v.view(B, N, self.n_heads, d_k)

        q = q + self.alpha_query.unsqueeze(0).unsqueeze(1)

        scores = torch.einsum('bnhd,bnhd->bnh', q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum('bnhd,bnh->bnhd', v, attn_weights)

        attn_output = attn_output.reshape(B, N, D)
        output = self.o_proj(attn_output)
        output = output + transformer_feat
        output = self.norm(output)

        return output


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)

        # Channel Transformer
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

        # 配置选项
        self.use_quantum_block = getattr(config, "use_quantum_block", False)
        self.use_attention_residuals = getattr(config, "use_attention_residuals", False)
        self.use_highway_gate = getattr(config, "use_highway_gate", False)

        n_heads = getattr(config, "n_heads", 4)

        if self.use_quantum_block:
            self.quantum_block = QuantumOTOCBlock(config.d_model, n_heads=n_heads)

            # 融合策略选择
            if self.use_highway_gate:
                self.highway_gate = HighwayGate(config.d_model)
                self.attn_residuals = None
            elif self.use_attention_residuals:
                self.attn_residuals = AttentionResiduals(config.d_model, n_heads=n_heads)
                self.highway_gate = None
            else:
                self.attn_residuals = None
                self.highway_gate = None
        else:
            self.quantum_block = None
            self.attn_residuals = None
            self.highway_gate = None

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

                if self.use_highway_gate and self.highway_gate is not None:
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
