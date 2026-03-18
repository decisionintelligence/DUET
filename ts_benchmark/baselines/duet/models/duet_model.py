from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import (
    Mahalanobis_mask,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)
import torch
import torch.nn.functional as F


class QuantumOTOCBlock(nn.Module):
    """
    Quantum-inspired feature mixing block operating on the feature dimension.

    This module follows your设计思路：
    - 将 encoder 输出视作复数态 psi_0
    - 通过 Cayley 变换构造幺正演化算符 U、测量基 M
    - 利用 OTOC 权重矩阵融合测量概率，得到 z_out

    只作用在最后一维 d_model 上，输入输出形状保持为 [B, N, D]。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 将实值特征映射到复数态的实部 / 虚部
        self.real_linear = nn.Linear(d_model, d_model)
        self.imag_linear = nn.Linear(d_model, d_model)

        # 哈密顿量 H 和测量算子 M 的可学习参数（实部和“虚部”权重）
        self.H_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.H_imag_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        self.M_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.M_imag_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)

    def _cayley_unitary(self, H: torch.Tensor) -> torch.Tensor:
        """
        通过 Cayley 变换得到幺正算符：
            U = (I - 0.5j H) (I + 0.5j H)^{-1}
        使用 torch.linalg.solve 避免显式求逆。
        """
        d = H.size(-1)
        I = torch.eye(d, device=H.device, dtype=H.dtype)

        A = I + 0.5j * H
        B = I - 0.5j * H
        # solve(A, B) 等价于 A^{-1} B
        U = torch.linalg.solve(A, B)
        return U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, D] 的实值特征
        :return: [B, N, D] 的实值特征（量子块输出）
        """
        B, N, D = x.shape

        # 1) 构造初始量子态 psi_0
        real = self.real_linear(x)
        imag = self.imag_linear(x)
        psi_0 = torch.complex(real, imag)  # [B, N, D]

        # 按特征维做归一化，避免数值发散
        norm = psi_0.abs().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        psi_0 = psi_0 / norm

        # 2) 构造哈密顿量 H，保持厄米性
        H_real = (self.H_weight + self.H_weight.t()) / 2
        H_imag = (self.H_imag_weight - self.H_imag_weight.t()) / 2
        H = torch.complex(H_real, H_imag)  # [D, D]

        # 控制谱半径，提升数值稳定性
        H = H / (H.abs().max().clamp_min(1e-3))

        # 3) 通过 Cayley 变换得到幺正演化算符 U
        U = self._cayley_unitary(H)  # [D, D]

        # 4) 演化 psi_0 -> psi_t
        psi_t = torch.matmul(psi_0, U.transpose(-2, -1))  # [B, N, D]

        # 5) 计算 OTOC 权重矩阵（基于 |U|^2）
        prob_matrix = torch.abs(U) ** 2  # [D, D]
        otoc_matrix = 2.0 * (prob_matrix * (1.0 - prob_matrix))
        otoc_weight = F.softmax(otoc_matrix, dim=-1)  # [D, D]

        # 6) 构造测量基 M，并对 psi_t 做测量
        M_real = (self.M_weight + self.M_weight.t()) / 2
        M_imag = (self.M_imag_weight - self.M_imag_weight.t()) / 2
        M_ham = torch.complex(M_real, M_imag)
        M_ham = M_ham / (M_ham.abs().max().clamp_min(1e-3))

        M_basis = self._cayley_unitary(M_ham)  # [D, D]
        psi_measured = torch.matmul(psi_t, M_basis)  # [B, N, D]

        # 概率幅：|psi_measured|^2，实值
        meas_prob = torch.abs(psi_measured) ** 2  # [B, N, D]

        # 7) 用 OTOC 权重做特征融合
        z_out = torch.matmul(meas_prob, otoc_weight.transpose(-2, -1))  # [B, N, D]
        z_out = self.norm(z_out)
        return z_out


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)

        # 原始 Channel Transformer，用于通道间关系建模
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

        # 量子 OTOC 块：仅在 config.use_quantum_block=True 时启用
        self.use_quantum_block = getattr(config, "use_quantum_block", False)
        if self.use_quantum_block:
            self.quantum_block = QuantumOTOCBlock(config.d_model)
        else:
            self.quantum_block = None

        self.linear_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.Dropout(config.fc_dropout),
        )

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, "b l n -> (b n) l 1")
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=input.shape[0]
            )
        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")

        if self.n_vars > 1:
            if self.quantum_block is not None:
                # 使用量子 OTOC 块在特征维上做混合
                channel_group_feature = self.quantum_block(temporal_feature)
            else:
                # 原始通道 Transformer 路径
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
