"""DX-PINN v3: BEAM decomposition + RGAF reliability + SE attention + Lite backbone

2026 innovations:
1. BEAM-Net decomposition (EMA trend/residual + spectral gating)
   - BEAM-Net: 2,835 params, 99.15% F1 @-6dB CWRU (Applied Sciences, Jun 2026)
2. RGAF-Net reliability weighting (sample-adaptive noise level fusion)
   - RGAF-Net: +19% F1 at -10dB vs WDCNN (Sensors, May 2026)
3. SE channel attention for noise suppression
   - SE-SDCTNet: 0.32M params, 93.1% strong noise (J.Vibroeng, 2026)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SAMPLE_LENGTH
from model.ccb import CausalConceptBottleneck
from model._base import ConvBNAct

HIDDEN_DIM = 64
DROPOUT = 0.15


# ===================== BEAM Decomposition =====================
class BEAMDenoiser(nn.Module):
    """BEAM-style denoiser: EMA decomposition + spectral gating.
    Compatible with existing training loop (returns x_denoised, noise_est, est_snr).
    """

    def __init__(self):
        super().__init__()
        n_freq = SAMPLE_LENGTH // 2 + 1
        hidden = max(16, n_freq // 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(hidden),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_freq),
            nn.Sigmoid(),
        )
        self.noise_est = nn.Sequential(
            nn.Conv1d(1, 8, 31, stride=8, padding=15),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        trend = F.avg_pool1d(x.unsqueeze(1), 7, 1, 3).squeeze(1)
        residual = x - trend

        spec = torch.fft.rfft(residual, dim=1)
        amp = torch.abs(spec)
        # Spectral gating via frequency-domain statistics
        amp_pooled = F.adaptive_avg_pool1d(amp.unsqueeze(1), amp.shape[1] // 8)
        g = self.gate(amp_pooled)  # [B, n_freq]
        amp_gated = amp * g
        spec_gated = torch.complex(
            amp_gated * torch.cos(torch.angle(spec)),
            amp_gated * torch.sin(torch.angle(spec)),
        )
        enhanced = torch.fft.irfft(spec_gated, n=SAMPLE_LENGTH, dim=1)
        x_denoised = trend + enhanced

        noise_level = self.noise_est(x.unsqueeze(1))
        est_snr = noise_level * 20 - 10  # [0,1] -> [-10,10]

        return x_denoised, noise_level, est_snr


# ===================== SE Attention =====================
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        h = max(4, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, h), nn.ReLU(),
            nn.Linear(h, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1)


# ===================== Multi-Scale Conv Stem =====================
class MultiScaleStem(nn.Module):
    def __init__(self, in_c=1, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        bd = hidden_dim // 4
        self.pre = ConvBNAct(in_c, bd, 15, stride=2, dropout=dropout)
        self.branches = nn.ModuleList([
            ConvBNAct(bd, bd, 3, dilation=d, dropout=dropout) for d in [1, 2, 4, 8]
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.MaxPool1d(4, 4),
        )
        self.sc = nn.Sequential(nn.Conv1d(bd, hidden_dim, 1), nn.MaxPool1d(4, 4))
        self.se = SEAttention(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.pre(x)
        ms = torch.cat([b(x) for b in self.branches], dim=1)
        return self.se(self.fuse(ms) + self.sc(x))


# ===================== Feature Extractor =====================
class FeatureExtractor(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 3, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 5, stride=2, dropout=dropout),
        )

    def forward(self, x):
        return self.conv(x).mean(dim=2)  # [B, C]


# ===================== RGAF Reliability Fusion =====================
class ReliabilityFusion(nn.Module):
    """Sample-adaptive fusion: higher noise → trust spectral features more."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, feat_trend, feat_res, reliability):
        return self.gate(torch.cat([feat_trend, feat_res, reliability], dim=1))


# ===================== DX-PINN v3 =====================
class DX_PINN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = BEAMDenoiser()
        self.stem_trend = MultiScaleStem(in_c=1)
        self.stem_res = MultiScaleStem(in_c=1)
        self.ext_t = FeatureExtractor()
        self.ext_r = FeatureExtractor()
        self.fusion = ReliabilityFusion()
        self.ccb = CausalConceptBottleneck(hidden_dim=HIDDEN_DIM)
        # Dummy heads for training compatibility
        self.preprocessor = nn.Identity()
        self.time_encoder = None
        self.freq_encoder = None
        self.cross_attention_fusion = None
        self.time_domain_head = None
        self.frequency_domain_head = None

    def forward(self, x, t=None):
        xd, noise_est, est_snr = self.denoiser(x)

        # Trend + residual decomposition
        trend = F.avg_pool1d(xd.unsqueeze(1), 7, 1, 3).squeeze(1)
        residual = xd - trend

        # Dual stems
        ft = self.stem_trend(trend)
        fr = self.stem_res(residual)

        # Feature extraction
        vt = self.ext_t(ft)
        vr = self.ext_r(fr)

        # Reliability-guided fusion
        fused = self.fusion(vt, vr, noise_est)

        # Classification
        ccb = self.ccb(fused)

        # Dummy values for compatibility
        B = x.shape[0]
        z = torch.zeros(B, SAMPLE_LENGTH, device=x.device)
        zd = torch.zeros(B, SAMPLE_LENGTH // 2, device=x.device)

        return {
            "x_denoised": xd, "noise_est": noise_est, "est_snr": est_snr,
            "cls_pred": ccb["cls_logits"], "c_hat": ccb["c_hat"],
            "domain_logits": ccb["domain_logits"],
            "z_causal": ccb["z_causal"], "z_context": ccb["z_context"],
            "shared_feature": fused, "time_feature": vt, "freq_feature": vr,
            "preprocessed_signal": xd, "raw_signal": x,
            "x_pred": z, "a_pred": z, "pred_amplitude": zd,
            "v_pred": z, "m": torch.zeros(B, 1, device=x.device),
            "c": torch.zeros(B, 1, device=x.device),
            "k": torch.zeros(B, 1, device=x.device),
        }
