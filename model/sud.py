"""Stockwell Unrolled Denoiser — 物理先验引导的可学习去噪前端"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import SAMPLE_LENGTH, BEARING_PARAMS


def sinc_kernel_init(num_kernels, kernel_size, freq_centers_hz, fs, device=None):
    """Sinc 函数族初始化卷积核，中心频率匹配轴承故障特征频带"""
    kernels = torch.zeros(num_kernels, 1, kernel_size)
    t = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size) / fs
    for i in range(num_kernels):
        fc = freq_centers_hz[i % len(freq_centers_hz)]
        h = 2 * fc * torch.sinc(2 * fc * t)
        h = h * torch.hamming_window(kernel_size)
        kernels[i, 0, :] = h / (h.std() + 1e-8)
    return kernels.to(device)


class SincConv1d(nn.Module):
    """Sinc 函数初始化的一维卷积层"""

    def __init__(self, out_channels, kernel_size, init_freqs, fs=12000):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        init_weights = sinc_kernel_init(out_channels, kernel_size, init_freqs, fs)
        self.conv.weight = nn.Parameter(init_weights)

    def forward(self, x):
        return self.conv(x)


class ISTALayer(nn.Module):
    """单层 ISTA 迭代软阈值去噪"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.W = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.Wt = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.lambda_thresh = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, y):
        residual = self.Wt(self.W(z) - y)
        z_new = z - self.alpha * residual
        return torch.sign(z_new) * F.relu(torch.abs(z_new) - F.softplus(self.lambda_thresh))


class SNREstimator(nn.Module):
    """轻量 SNR 估计器，自适应控制去噪深度"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(64, 8, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(8 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * 15 - 5


class StockwellUnrolledDenoiser(nn.Module):
    """Stockwell 展开式可学习去噪器

    输入: noisy_signal [B, L]
    输出: denoised [B, L], noise_est [B, 1], est_snr [B, 1]
    """

    def __init__(self, num_kernels=128, kernel_size=15, num_ista_layers=5, fs=BEARING_PARAMS["fs"]):
        super().__init__()
        self.fs = fs
        self.num_ista_layers = num_ista_layers
        self.num_kernels = num_kernels

        fault_freqs = self._compute_fault_freq_centers()
        self.sinc_conv = SincConv1d(num_kernels, kernel_size, fault_freqs, fs)

        self.ista_layers = nn.ModuleList([
            ISTALayer(num_kernels, kernel_size=5) for _ in range(num_ista_layers)
        ])

        self.reconstruct = nn.Conv1d(num_kernels, 1, kernel_size=7, padding=3)
        # Dynamic SNREstimator based on num_kernels
        self.snr_estimator = nn.Sequential(
            nn.Conv1d(num_kernels, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(16 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.noise_estimator = nn.Sequential(
            nn.Conv1d(num_kernels, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def _compute_fault_freq_centers(self):
        """计算 CWRU 轴承的故障特征频率中心"""
        from utils import calculate_fault_frequency
        freqs = []
        for wc in [0, 1, 2, 3]:
            ff = calculate_fault_frequency(wc)
            for key in ["bpfi", "bpfo", "bsf", "fr"]:
                for k in range(1, 5):
                    freqs.append(ff[key] * k)
        return sorted(set(freqs))

    def forward(self, x):
        x_in = x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, L]
        z = self.sinc_conv(x_in)  # [B, K, L]
        y = z.clone()
        est_snr = self.snr_estimator(z)

        # ISTA unrolled denoising with skip connections
        for idx, layer in enumerate(self.ista_layers):
            z_new = layer(z, y)
            z = z + z_new  # residual connection for better gradient flow

        # Predict clean signal directly
        denoised = self.reconstruct(z).squeeze(1)
        noise_est = self.noise_estimator(z)
        return denoised, noise_est, est_snr
