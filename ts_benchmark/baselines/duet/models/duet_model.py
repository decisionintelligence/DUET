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
    Quantum-inspired feature mixing block 

    改进点：
    1. 多 head 细粒度混合：利用 n_heads 将特征分成多组，每组独立进行 OTOC 演化
    2. 残差连接：保留原始特征，避免信息丢失
    3. 改进的 Hamiltonian 初始化：从单位矩阵开始，更稳定

    输入输出形状：[B, N, D] -> [B, N, D]
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 将实值特征映射到复数态的实部 / 虚部
        self.real_linear = nn.Linear(d_model, d_model)
        self.imag_linear = nn.Linear(d_model, d_model)

        # 多 head 的哈密顿量 H 和测量算子 M（按你的注释思路）
        # 形状: [n_heads, d_k, d_k]
        nn_init = nn.init.eye_
        self.H_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        self.H_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        # 用单位矩阵初始化实部
        for i in range(n_heads):
            nn_init(self.H_real.data[i])
            nn.init.zeros_(self.H_imag.data[i])

        self.M_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        self.M_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.01)
        for i in range(n_heads):
            nn_init(self.M_real.data[i])
            nn_init(self.M_imag.data[i])

        # 融合后的投影层
        self.projection = nn.Linear(d_model, d_model)

        # 残差连接的可学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(d_model)

    def _cayley_unitary(self, H: torch.Tensor) -> torch.Tensor:
        """通过 Cayley 变换得到幺正算符"""
        d = H.size(-1)
        I = torch.eye(d, device=H.device, dtype=H.dtype)
        A = I + 0.5j * H
        B = I - 0.5j * H
        U = torch.linalg.solve(A, B)
        return U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # 保存原始输入用于残差连接
        x_residual = x

        # 1) 构造初始量子态 psi_0
        real = self.real_linear(x)
        imag = self.imag_linear(x)
        psi_0 = torch.complex(real, imag)

        # 归一化
        norm = psi_0.abs().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        psi_0 = psi_0 / norm

        # 2) 变换到多 head 视角: [B, N, D] -> [B, N, n_heads, d_k]
        psi_0 = psi_0.view(B, N, self.n_heads, self.d_k)

        # 3) 获取多 head 的幺正演化算符 U 和测量基 M
        H_real = (self.H_real + self.H_real.transpose(-2, -1)) / 2
        H_imag = (self.H_imag - self.H_imag.transpose(-2, -1)) / 2
        H = torch.complex(H_real, H_imag)

        M_real = (self.M_real + self.M_real.transpose(-2, -1)) / 2
        M_imag = (self.M_imag - self.M_imag.transpose(-2, -1)) / 2
        M_ham = torch.complex(M_real, M_imag)

        # 控制谱半径
        H = H / (H.abs().max().clamp_min(1e-3))
        M_ham = M_ham / (M_ham.abs().max().clamp_min(1e-3))

        U = self._cayley_unitary(H)
        M_basis = self._cayley_unitary(M_ham)

        # 4) 量子演化 psi_0 -> psi_t
        psi_t = torch.einsum('bnhd,hde->bnhe', psi_0, U.conj().transpose(-2, -1))

        # 5) 测量
        psi_measured = torch.einsum('bnhd,hde->bnhe', psi_t, M_basis)
        meas_prob = torch.abs(psi_measured) ** 2

        # 6) 计算 OTOC 权重矩阵
        prob_matrix = torch.abs(U) ** 2
        otoc_matrix = 2.0 * (prob_matrix * (1.0 - prob_matrix))
        otoc_weight = F.softmax(otoc_matrix, dim=-1)

        # 7) 用 OTOC 权重做特征融合
        z_out = torch.einsum('bnhd,hde->bnhe', meas_prob, otoc_weight.transpose(-2, -1))

        # 8) 还原维度: [B, N, n_heads, d_k] -> [B, N, D]
        z_out = z_out.reshape(B, N, D)
        z_out = self.norm(z_out)

        # 9) 投影
        z_out = self.projection(z_out)

        # 10) 残差连接：融合原始特征和量子块输出
        z_out = self.alpha * z_out + (1 - self.alpha) * x_residual

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
            # 传入 n_heads 参数
            n_heads = getattr(config, "n_heads", 4)
            self.quantum_block = QuantumOTOCBlock(config.d_model, n_heads=n_heads)
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
