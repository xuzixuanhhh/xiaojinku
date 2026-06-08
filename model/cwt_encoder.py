"""CWT-based Frequency Encoder — noise-robust TF analysis

Replaces FFT with differentiable CWT (Morlet wavelets).
CWT creates sparse TF representations naturally noise-resistant.

Ref: SCBM-Net (Nature SR, 2025): CWT+Swin achieves 80.67% at -10dB
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import ENCODER_HIDDEN_DIM, SAMPLE_LENGTH, BEARING_PARAMS


def morlet_wavelet_kernel(scale, fs, kernel_size=256):
    t = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size) / fs
    w0 = 6.0
    real = torch.cos(w0 * t / scale) * torch.exp(-t ** 2 / (2 * scale ** 2))
    return real / (real.std() + 1e-8)


class CWTConv1d(nn.Module):
    """Differentiable CWT via Conv1d with Morlet kernels"""

    def __init__(self, n_scales=64, kernel_size=256, fs=BEARING_PARAMS["fs"]):
        super().__init__()
        self.n_scales = n_scales
        min_freq = 20
        max_freq = fs / 4
        scales = fs / (2 * np.pi * np.geomspace(min_freq, max_freq, n_scales))
        kernels = []
        for s in scales:
            k = morlet_wavelet_kernel(s, fs, kernel_size)
            kernels.append(k)
        kernels = torch.stack(kernels).unsqueeze(1)
        self.register_buffer('wavelet_kernels', kernels)
        self.pad = kernel_size // 2

    def forward(self, x):
        B = x.shape[0]
        cwt_out = F.conv1d(x.unsqueeze(1), self.wavelet_kernels, padding=self.pad)
        return torch.log1p(torch.abs(cwt_out))


class CWTEncoder(nn.Module):
    """CWT-based noise-robust encoder. Replaces FrequencyDomainEncoder."""

    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, n_scales=64, dropout=0.2):
        super().__init__()
        self.cwt = CWTConv1d(n_scales=n_scales)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_scales, hidden_dim // 2, 7, 2, 3),
            nn.BatchNorm1d(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 5, 2, 2),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 5, 2, 2),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        cwt_out = self.cwt(x)
        seq = self.cnn(cwt_out).transpose(1, 2)
        seq = self.transformer(seq)
        seq = self.norm(seq)
        return seq, seq.mean(dim=1)
