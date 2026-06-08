"""DX-PINN v2: DRSN adaptive thresholding + sparse attention + physics kernels

Key improvements (from 2024-2025 literature):
1. DRSN-style Adaptive Residual Shrinkage Blocks (replace fixed ISTA)
   - Per-channel learnable thresholds via GAP->FC->Sigmoid
   - Reference: DRSN (Deep Residual Shrinkage Network), DRSwin-ST (2024)
2. Sparse attention (1.5-entmax) in transformer encoders
   - Produces sparser attention, naturally suppresses noise
   - Reference: DRSwin-ST (Reliability Engineering & System Safety, 2024)
3. Physics-informed SincConv initialization (kept from original)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SAMPLE_LENGTH, BEARING_PARAMS
from model.ccb import CausalConceptBottleneck
from model._base import (
    RobustSignalPreprocessor, CrossAttentionFusion,
    TimeDomainHead, FrequencyDomainHead,
    ConvBNAct,
)

HIDDEN_DIM = 64
ENC_LAYERS = 4
FREQ_LAYERS = 2
NUM_KERNELS = 64
DROPOUT = 0.15


# ===================== Sparse Attention (alpha-entmax) =====================
def entmax15(logits, dim=-1, n_iter=20):
    """1.5-entmax: produces sparse probability distributions.
    Generalizes softmax(alpha=1) and sparsemax(alpha=2).
    Reference: DRSwin-ST (Zhou et al., 2024) — replaces Softmax to suppress noise.
    """
    alpha = 1.5
    # Bisection on tau for the alpha-entmax projection
    # p_i = [(alpha-1)*tau - logits_i]_+ ^ (1/(alpha-1))
    # Since alpha=1.5, 1/(alpha-1) = 2, so p_i = relu(0.5*tau - logits_i)^2
    tau_min = logits.max(dim=dim, keepdim=True).values - 10.0
    tau_max = logits.max(dim=dim, keepdim=True).values + 10.0

    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        p = torch.relu(tau * 0.5 - logits).pow(2)
        s = p.sum(dim=dim, keepdim=True)
        # We want sum(p) ≈ 1
        tau_min = torch.where(s > 1.0, tau, tau_min)
        tau_max = torch.where(s <= 1.0, tau, tau_max)

    tau = (tau_min + tau_max) / 2
    p = torch.relu(tau * 0.5 - logits).pow(2)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-8)


# ===================== DRSN Adaptive Threshold =====================
class AdaptiveChannelThreshold(nn.Module):
    """Per-channel learnable soft-threshold (DRSN-style).
    GAP -> FC -> ReLU -> FC -> Sigmoid -> scaled per-channel threshold.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        scales = self.net(x)  # [B, C]
        mean_abs = x.abs().mean(dim=2)  # [B, C]
        thresholds = (scales * mean_abs * 0.8).unsqueeze(-1)
        return torch.sign(x) * F.relu(x.abs() - thresholds)


class AdaptiveShrinkBlock(nn.Module):
    """DRSN residual block: Conv->BN->GELU->Conv->BN->AdaptiveThreshold + Residual"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.shrink = AdaptiveChannelThreshold(channels)

    def forward(self, x):
        r = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shrink(out)
        return F.gelu(out + r)


# ===================== SincConv (physics-informed kernels) =====================
class SincConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, fs=12000):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        from model.sud import sinc_kernel_init
        from utils import calculate_fault_frequency
        freqs = []
        for wc in [0, 1, 2, 3]:
            ff = calculate_fault_frequency(wc)
            for key in ["bpfi", "bpfo", "bsf", "fr"]:
                for k in range(1, 5):
                    freqs.append(ff[key] * k)
        centers = sorted(set(freqs))
        w = sinc_kernel_init(out_channels, kernel_size, centers, fs)
        self.conv.weight = nn.Parameter(w)

    def forward(self, x):
        return self.conv(x)


# ===================== Adaptive Denoiser (DRSN-based) =====================
class AdaptiveDenoiser(nn.Module):
    """Replace SUD's ISTA layers with DRSN Adaptive Shrinkage Blocks."""

    def __init__(self, num_kernels=NUM_KERNELS, num_blocks=4):
        super().__init__()
        self.sinc_conv = SincConv1d(num_kernels, 15)
        self.blocks = nn.ModuleList([
            AdaptiveShrinkBlock(num_kernels, kernel_size=5)
            for _ in range(num_blocks)
        ])
        self.recon = nn.Conv1d(num_kernels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.sinc_conv(x)
        for b in self.blocks:
            z = b(z)
        d = self.recon(z).squeeze(1)
        return d, torch.zeros(x.shape[0], 1, device=x.device), torch.zeros(x.shape[0], 1, device=x.device)


# ===================== Sparse Transformer Layer =====================
class SparseTransformerLayer(nn.Module):
    """Transformer with 1.5-entmax sparse attention."""

    def __init__(self, d_model, nhead=4, dim_ff=None, dropout=0.1):
        super().__init__()
        dim_ff = dim_ff or d_model * 2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Sparse self-attention with pre-norm
        r = x
        x = self.norm1(x)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = entmax15(attn, dim=-1)  # sparse attention
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = r + self.out_proj(out)
        return out + self.ff(self.norm2(out))


# ===================== Encoders =====================
class TimeEncoderV2(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=ENC_LAYERS, dropout=DROPOUT):
        super().__init__()
        bd = hidden_dim // 4
        self.pre = ConvBNAct(1, bd, kernel_size=15, stride=2, dropout=dropout)
        self.branches = nn.ModuleList([
            ConvBNAct(bd, bd, 3, dilation=d, dropout=dropout)
            for d in [1, 2, 4, 8]
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.MaxPool1d(4, 4),
        )
        self.sc = nn.Sequential(nn.Conv1d(bd, hidden_dim, 1), nn.MaxPool1d(4, 4))
        self.transformers = nn.ModuleList([
            SparseTransformerLayer(hidden_dim, nhead=4, dropout=dropout)
            for _ in range(max(1, num_layers // 2))
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.pre(x.unsqueeze(1))
        ms = torch.cat([b(x) for b in self.branches], dim=1)
        seq = (self.fuse(ms) + self.sc(x)).transpose(1, 2)
        for t in self.transformers:
            seq = t(seq)
        feat = self.norm(seq).mean(dim=1)
        return seq, feat


class FreqEncoderV2(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=FREQ_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBNAct(1, hidden_dim // 2, 9, stride=2, dropout=dropout),
            ConvBNAct(hidden_dim // 2, hidden_dim, 7, stride=2, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 5, stride=2, dropout=dropout),
        )
        self.transformers = nn.ModuleList([
            SparseTransformerLayer(hidden_dim, nhead=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        s = torch.fft.rfft(x, dim=1)
        amp = torch.abs(s[:, 1:SAMPLE_LENGTH // 2 + 1])
        nf = amp.median(dim=1, keepdim=True).values
        amp = torch.log1p(amp / (nf + 1e-6))
        amp = (amp - amp.mean(dim=1, keepdim=True)) / (amp.std(dim=1, keepdim=True) + 1e-6)
        seq = self.cnn(amp.unsqueeze(1)).transpose(1, 2)
        for t in self.transformers:
            seq = t(seq)
        feat = self.norm(seq).mean(dim=1)
        return seq, feat


# ===================== DX-PINN v2 =====================
class DX_PINN_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = AdaptiveDenoiser()
        self.preprocessor = RobustSignalPreprocessor()
        self.time_encoder = TimeEncoderV2()
        self.freq_encoder = FreqEncoderV2()
        self.cross_attention_fusion = CrossAttentionFusion(hidden_dim=HIDDEN_DIM)
        self.time_domain_head = TimeDomainHead(hidden_dim=HIDDEN_DIM)
        self.frequency_domain_head = FrequencyDomainHead(hidden_dim=HIDDEN_DIM)
        self.ccb = CausalConceptBottleneck(hidden_dim=HIDDEN_DIM)

    def forward(self, x, t):
        xd, ne, es = self.denoiser(x)
        xp = self.preprocessor(xd)
        ts, tf = self.time_encoder(xp)
        fs, ff = self.freq_encoder(xd)
        fused, tf2, ff2 = self.cross_attention_fusion(ts, fs)
        ccb = self.ccb(fused)
        x_pred, a_pred, pinn = self.time_domain_head(fused, t)
        pa = self.frequency_domain_head(fused)

        return {
            "x_denoised": xd, "noise_est": ne, "est_snr": es,
            "cls_pred": ccb["cls_logits"], "c_hat": ccb["c_hat"],
            "domain_logits": ccb["domain_logits"],
            "z_causal": ccb["z_causal"], "z_context": ccb["z_context"],
            "shared_feature": fused,
            "time_feature": tf, "freq_feature": ff,
            "preprocessed_signal": xp, "raw_signal": x,
            "x_pred": x_pred, "a_pred": a_pred,
            "pred_amplitude": pa,
            "v_pred": pinn["v_pred"],
            "m": pinn["m"], "c": pinn["c"], "k": pinn["k"],
        }
