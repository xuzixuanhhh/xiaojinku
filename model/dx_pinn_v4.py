"""DX-PINN v4: ECA + Selective Kernel + Dynamic Gate + Balanced ~1.2M params

2026 modules:
1. ECA (Efficient Channel Attention) — MSF-ECA, ~40 params, 1D conv cross-channel
2. Selective Kernel Fusion — multi-scale branches with learned attention weights
3. Dynamic Gating — C-SwinNet style input-dependent routing
4. Large Kernel Depthwise Conv — SLKCResNet style wide receptive field
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SAMPLE_LENGTH
from model.ccb import CausalConceptBottleneck
from model._base import ConvBNAct

HIDDEN_DIM = 96
DROPOUT = 0.15
NUM_KERNELS = 96


class ECA(nn.Module):
    """Efficient Channel Attention — 1D conv instead of FC, ~40 params."""

    def __init__(self, channels, k=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).unsqueeze(1)
        y = self.conv(y).squeeze(1).unsqueeze(-1)
        return x * self.sigmoid(y)


class LargeKernelDWConv(nn.Module):
    """Depthwise-separable large kernel conv — wide field, few params."""

    def __init__(self, channels, kernel_size=15):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                             padding=kernel_size // 2, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return F.gelu(self.bn(self.pw(self.dw(x))))


class SelectiveKernelFusion(nn.Module):
    """Multi-scale branches with learned attention weights per kernel scale."""

    def __init__(self, channels, kernels=[3, 5, 7, 15]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                LargeKernelDWConv(channels, k),
            ) for k in kernels
        ])
        self.attn_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, channels // 4), nn.ReLU(),
            nn.Linear(channels // 4, len(kernels)), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        outputs = [b(x) for b in self.branches]
        stacked = torch.stack(outputs, dim=2)
        w = self.attn_fc(x).unsqueeze(1).unsqueeze(-1)
        return (stacked * w).sum(dim=2) + x


class LightDenoiser(nn.Module):
    """Lightweight denoiser: SincConv + SK blocks + ECA."""

    def __init__(self, num_kernels=NUM_KERNELS):
        super().__init__()
        from model.sud import sinc_kernel_init
        from utils import calculate_fault_frequency
        freqs = []
        for wc in [0, 1, 2, 3]:
            ff = calculate_fault_frequency(wc)
            for key in ["bpfi", "bpfo", "bsf", "fr"]:
                for k in range(1, 5):
                    freqs.append(ff[key] * k)
        centers = sorted(set(freqs))

        self.sinc_conv = nn.Conv1d(1, num_kernels, 15, padding=7, bias=False)
        w = sinc_kernel_init(num_kernels, 15, centers, 12000)
        self.sinc_conv.weight = nn.Parameter(w)

        self.sk1 = SelectiveKernelFusion(num_kernels)
        self.eca1 = ECA(num_kernels)
        self.sk2 = SelectiveKernelFusion(num_kernels)
        self.eca2 = ECA(num_kernels)
        self.lk = LargeKernelDWConv(num_kernels, kernel_size=31)
        self.recon = nn.Conv1d(num_kernels, 1, 7, padding=3)
        self.noise_est = nn.Sequential(
            nn.Conv1d(num_kernels, 16, 1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.sinc_conv(x)
        z = self.eca1(self.sk1(z))
        z = self.eca2(self.sk2(z))
        z = z + self.lk(z)
        d = self.recon(z).squeeze(1)
        ne = self.noise_est(z)
        return d, ne, ne * 20 - 10


class MultiScaleStem(nn.Module):
    def __init__(self, in_c=1, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        bd = hidden_dim // 4
        self.pre = ConvBNAct(in_c, bd, 15, stride=2, dropout=dropout)
        self.branches = nn.ModuleList([
            ConvBNAct(bd, bd, k, dilation=d, dropout=dropout)
            for k, d in [(3, 1), (5, 2), (7, 4), (9, 8)]
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.MaxPool1d(4, 4),
        )
        self.sc = nn.Sequential(nn.Conv1d(bd, hidden_dim, 1), nn.MaxPool1d(4, 4))
        self.eca = ECA(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.pre(x)
        ms = torch.cat([b(x) for b in self.branches], dim=1)
        return self.eca(self.fuse(ms) + self.sc(x))


class TimeFeatExtractor(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 3, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 5, stride=2, dropout=dropout),
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.eca = ECA(hidden_dim)

    def forward(self, x):
        seq = self.conv(x).transpose(1, 2)
        seq = self.norm(self.tf(seq))
        seq = self.eca(seq.transpose(1, 2)).transpose(1, 2)
        return seq, seq.mean(dim=1)


class FreqFeatExtractor(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=1, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBNAct(1, hidden_dim // 2, 9, stride=2, dropout=dropout),
            ConvBNAct(hidden_dim // 2, hidden_dim, 7, stride=2, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, 5, stride=2, dropout=dropout),
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        s = torch.fft.rfft(x, dim=1)
        amp = torch.abs(s[:, 1:SAMPLE_LENGTH // 2 + 1])
        nf = amp.median(dim=1, keepdim=True).values
        amp = torch.log1p(amp / (nf + 1e-6))
        amp = (amp - amp.mean(dim=1, keepdim=True)) / (amp.std(dim=1, keepdim=True) + 1e-6)
        seq = self.cnn(amp.unsqueeze(1)).transpose(1, 2)
        return seq, self.norm(self.tf(seq)).mean(dim=1)


class DynamicGate(nn.Module):
    """Input-dependent gated fusion between time and frequency features."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
        )

    def forward(self, ft, ff):
        g = self.gate(torch.cat([ft, ff], dim=1))
        routed = g * ft + (1 - g) * ff
        return self.fusion(torch.cat([routed, (ft + ff) * 0.5], dim=1))


class DX_PINN_V4(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = LightDenoiser()
        self.stem_t = MultiScaleStem(in_c=1)
        self.stem_f = MultiScaleStem(in_c=1)
        self.ext_t = TimeFeatExtractor()
        self.ext_f = FreqFeatExtractor()
        self.dy_gate = DynamicGate()
        self.ccb = CausalConceptBottleneck(hidden_dim=HIDDEN_DIM)
        self.preprocessor = nn.Identity()
        self.time_encoder = None; self.freq_encoder = None
        self.cross_attention_fusion = None
        self.time_domain_head = None; self.frequency_domain_head = None

    def forward(self, x, t=None):
        xd, ne, es = self.denoiser(x)
        st = self.stem_t(xd)
        sf = self.stem_f(xd)
        seq_t, ft = self.ext_t(st)
        seq_f, ff = self.ext_f(xd)
        fused = self.dy_gate(ft, ff)
        ccb = self.ccb(fused)

        B = x.shape[0]; z = torch.zeros(B, SAMPLE_LENGTH, device=x.device)
        return {
            "x_denoised": xd, "noise_est": ne, "est_snr": es,
            "cls_pred": ccb["cls_logits"], "c_hat": ccb["c_hat"],
            "domain_logits": ccb["domain_logits"],
            "z_causal": ccb["z_causal"], "z_context": ccb["z_context"],
            "shared_feature": fused, "time_feature": ft, "freq_feature": ff,
            "preprocessed_signal": xd, "raw_signal": x,
            "x_pred": z, "a_pred": z,
            "pred_amplitude": torch.zeros(B, SAMPLE_LENGTH // 2, device=x.device),
            "v_pred": z, "m": torch.zeros(B, 1, device=x.device),
            "c": torch.zeros(B, 1, device=x.device),
            "k": torch.zeros(B, 1, device=x.device),
        }
