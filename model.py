import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.Ipu = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.R = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.delta_theta_fl = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.tau_o = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.k_t = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.load_constant1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.load_constant2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.router_linear = nn.Sequential(
            nn.Linear(24, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.physics_loss_weights = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))


        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model//2, bias=True),
            nn.ReLU(),
            nn.Linear(d_model//2, c_out, bias=True)
        )

        self.H_weight = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.H_imag_weight = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.norm = nn.LayerNorm(d_model)
        self.M_weight = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.M_imag_weight = nn.Parameter(torch.randn(d_model, d_model), requires_grad=True)
        self.iamg_linear = nn.Linear(d_model, d_model)
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 静态 Hamiltonian 参数 [H, d_k, d_k]
        self.H_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k), requires_grad=True)
        self.H_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k), requires_grad=True)
        
        # 静态可学习测量基参数 [H, d_k, d_k]
        self.M_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k), requires_grad=True)
        self.M_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k), requires_grad=True)

        self.cancha_1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.cancha_2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        
    def _router(self, x):
        flatten = x.view(x.shape[0], -1)  # shape: [32, 24]
        output = self.router_linear(flatten)  # shape: [32, 2]

        # 使用softmax确保 g1 + g2 = 1
        gates = torch.nn.functional.softmax(output, dim=-1)
        g1, g2, g3 = gates[:, 0], gates[:, 1], gates[:, 2]
        return g1, g2, g3
    
    def _get_unitary(self, real, imag):
        """Cayley Transform: U = (I - 0.5j*H)(I + 0.5j*H)^-1"""
        # 确保 Hermitian
        H = torch.complex(real, imag)
        H = (H + H.conj().transpose(-2, -1)) / 2
        
        I = torch.eye(self.d_k, device=real.device).unsqueeze(0) # [1, d_k, d_k]
        U = torch.matmul(I - 0.5j * H, torch.inverse(I + 0.5j * H))
        return U
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # print("Input encoder shape:", x_enc.shape)  # Debugging line torch.Size([32, 96, 321])

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print("Encoder embedding output shape:", enc_out.shape)  # Debugging line torch.Size([32, 96, 512])
        
        # 初始态 psi_0 的构造：将实部和虚部结合成一个复数张量
        # enc_out_iamg = self.iamg_linear(enc_out)
        enc_out = self.norm(enc_out)
        psi_0 = torch.complex(enc_out, enc_out) / torch.sqrt(torch.tensor(2.0))


        # 构造哈密顿量 H，保持其厄米性质
        H_real = (self.H_weight + self.H_weight.t()) / 2
        H_imag = (self.H_imag_weight - self.H_imag_weight.t()) / 2
        Hamitonian = torch.complex(H_real, H_imag) # [D, D]
        I = torch.eye(Hamitonian.size(0), device=Hamitonian.device, dtype=Hamitonian.dtype)
        Unitary_Operator = torch.matmul(I - 0.5j * Hamitonian, torch.inverse(I + 0.5j * Hamitonian))

        # 演化 psi_0 得到 psi_t
        psi_t = torch.matmul(psi_0, Unitary_Operator.t())
        # print("Shape of psi_t:", psi_t.shape)  # Debugging line torch.Size([32, 96, 512])

        # 计算 OTOC 权重矩阵
        OTOC_matrix = torch.abs(Unitary_Operator) ** 2
        OTOC_weight = F.softmax(OTOC_matrix, dim=-1)

        # 定义一个可学习的测量算子 M (同样保持幺正性)
        M_real = (self.M_weight + self.M_weight.t()) / 2
        M_imag = (self.M_imag_weight - self.M_imag_weight.t()) / 2
        M_ham = torch.complex(M_real, M_imag)
        M_basis = torch.matmul(I - 0.5j * M_ham, torch.inverse(I + 0.5j * M_ham))
        psi_measured = torch.matmul(psi_t, M_basis)
        meas_prob = torch.abs(psi_measured) ** 2

        z_out = torch.matmul(meas_prob, OTOC_weight.t())
        z_out = self.norm(z_out)
        # print("Shape of z_out:", z_out.shape)  # Debugging line torch.Size([32, 96, 512])

        
        '''
        B, L, D = enc_out.shape
        psi_0 = psi_0.view(B, L, self.n_heads, self.d_k)  # [B, L, n_heads, d_k]
        # 1. 获取静态幺正算符和测量基
        U = self._get_unitary(self.H_real, self.H_imag) # [n_heads, d_k, d_k]
        M = self._get_unitary(self.M_real, self.M_imag) # [n_heads, d_k, d_k]
        psi_t = torch.einsum('blhd, hde -> blhe', psi_0, U.transpose(-2, -1))
        psi_measured = torch.einsum('blhd, hde -> blhe', psi_t, M)
        meas_prob = torch.abs(psi_measured) ** 2 # [B, L, n_heads, d_k]

        prob_matrix = torch.abs(U) ** 2 # [n_heads, d_k, d_k]
        otoc_matrix = 2 * (prob_matrix * (1 - prob_matrix))
        otoc_weight = F.softmax(otoc_matrix, dim=-1) # [n_heads, d_k, d_k]
        
        # 5. 维度内特征融合
        # z = meas_prob @ otoc_weight.T
        z_out = torch.einsum('blhd, hde -> blhe', meas_prob, otoc_weight.transpose(-2, -1))
        
        # 6. 还原维度并投影
        z_out = z_out.reshape(B, L, D)
        z_out = self.norm(z_out)
        '''

        
        enc_out, attns = self.encoder(z_out, attn_mask=enc_self_mask)
        # print("Encoder output shape:", enc_out.shape)  # Debugging line torch.Size([32, 48, 512])
        enc_out = self.cancha_1 * enc_out + self.cancha_2 * z_out[:, :enc_out.size(1), :]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # print("Decoder embedding output shape:", dec_out.shape)  # Debugging line torch.Size([32, 72, 512])

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # print("Decoder output shape before projection:", dec_out.shape)  # Debugging line torch.Size([32, 72, 512])

        dec_out = self.projection(dec_out)

        # print("Decoder output shape after projection:", dec_out.shape)  # Debugging line torch.Size([32, 72, 1])
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:], OTOC_matrix # [B, L, D]
