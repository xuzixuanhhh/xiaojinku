"""DX-PINN v5: Wider ECA+SK+DyGate — HIDDEN_DIM=128, ~1.5M params"""
import torch, torch.nn as nn, torch.nn.functional as F
from config import SAMPLE_LENGTH
from model.ccb import CausalConceptBottleneck
from model._base import ConvBNAct

HD = 128; DP = 0.15; NK = 128


class ECA(nn.Module):
    def __init__(self, c, k=5):
        super().__init__(); self.g = nn.AdaptiveAvgPool1d(1)
        self.c = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        y = self.g(x).squeeze(-1).unsqueeze(1)
        return x * torch.sigmoid(self.c(y).squeeze(1).unsqueeze(-1))


class LKDW(nn.Module):
    def __init__(self, c, k=15):
        super().__init__()
        self.d = nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.p = nn.Conv1d(c, c, 1, bias=False); self.b = nn.BatchNorm1d(c)
    def forward(self, x): return F.gelu(self.b(self.p(self.d(x))))


class SK(nn.Module):
    def __init__(self, c, ks=[3, 5, 7, 15, 31]):
        super().__init__()
        self.b = nn.ModuleList([LKDW(c, k) for k in ks])
        self.a = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                               nn.Linear(c, c // 4), nn.ReLU(),
                               nn.Linear(c // 4, len(ks)), nn.Softmax(dim=-1))
    def forward(self, x):
        o = torch.stack([b(x) for b in self.b], dim=2)
        w = self.a(x).unsqueeze(1).unsqueeze(-1)
        return (o * w).sum(dim=2) + x


class Denoiser(nn.Module):
    def __init__(self, nk=NK):
        super().__init__()
        from model.sud import sinc_kernel_init
        from utils import calculate_fault_frequency
        from config import BEARING_PARAMS
        fs = BEARING_PARAMS["fs"]
        freqs = []
        for wc in [0, 1, 2, 3]:
            ff = calculate_fault_frequency(wc)
            for key in ["bpfi", "bpfo", "bsf", "fr"]:
                for k in range(1, 5): freqs.append(ff[key] * k)
        c = sorted(set(freqs))
        ks = max(15, int(15 * fs / 12000))
        if ks % 2 == 0: ks += 1  # ensure odd for same-padding
        self.s = nn.Conv1d(1, nk, ks, padding=ks // 2, bias=False)
        self.s.weight = nn.Parameter(sinc_kernel_init(nk, ks, c, fs))
        self.k1 = SK(nk); self.e1 = ECA(nk)
        self.k2 = SK(nk); self.e2 = ECA(nk)
        self.l = LKDW(nk, 31); self.r = nn.Conv1d(nk, 1, 7, padding=3)
        self.n = nn.Sequential(nn.Conv1d(nk, 16, 1), nn.ReLU(),
                                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        z = self.s(x); z = self.e1(self.k1(z)); z = self.e2(self.k2(z))
        z = z + self.l(z); d = self.r(z).squeeze(1); ne = self.n(z)
        return d, ne, ne * 20 - 10


class Stem(nn.Module):
    def __init__(self, ic=1, hd=HD, dp=DP):
        super().__init__(); bd = hd // 4
        self.p = ConvBNAct(ic, bd, 15, stride=2, dropout=dp)
        self.b = nn.ModuleList([ConvBNAct(bd, bd, k, dilation=d, dropout=dp)
                                 for k, d in [(3, 1), (5, 2), (7, 4), (9, 8)]])
        self.f = nn.Sequential(nn.Conv1d(hd, hd, 1), nn.BatchNorm1d(hd),
                                nn.GELU(), nn.MaxPool1d(4, 4))
        self.s = nn.Sequential(nn.Conv1d(bd, hd, 1), nn.MaxPool1d(4, 4))
        self.e = ECA(hd)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        elif x.shape[1] != 1: x = x.mean(dim=1, keepdim=True)
        x = self.p(x)
        return self.e(self.f(torch.cat([b(x) for b in self.b], dim=1)) + self.s(x))


class TExt(nn.Module):
    def __init__(self, hd=HD, nl=3, dp=DP):
        super().__init__()
        self.c = nn.Sequential(ConvBNAct(hd, hd, 3, dropout=dp),
                                ConvBNAct(hd, hd, 3, dropout=dp),
                                ConvBNAct(hd, hd, 5, stride=2, dropout=dp))
        e = nn.TransformerEncoderLayer(d_model=hd, nhead=4, dim_feedforward=hd * 4,
                                        dropout=dp, activation='gelu', batch_first=True, norm_first=True)
        self.t = nn.TransformerEncoder(e, num_layers=nl)
        self.n = nn.LayerNorm(hd); self.e = ECA(hd)
    def forward(self, x):
        s = self.c(x).transpose(1, 2); s = self.n(self.t(s))
        return self.e(s.transpose(1, 2)).transpose(1, 2), s.mean(dim=1)


class FExt(nn.Module):
    def __init__(self, hd=HD, nl=2, dp=DP):
        super().__init__()
        self.c = nn.Sequential(ConvBNAct(1, hd // 2, 9, stride=2, dropout=dp),
                                ConvBNAct(hd // 2, hd, 7, stride=2, dropout=dp),
                                ConvBNAct(hd, hd, 5, stride=2, dropout=dp))
        e = nn.TransformerEncoderLayer(d_model=hd, nhead=4, dim_feedforward=hd * 2,
                                        dropout=dp, activation='gelu', batch_first=True, norm_first=True)
        self.t = nn.TransformerEncoder(e, num_layers=nl); self.n = nn.LayerNorm(hd)
    def forward(self, x):
        s = torch.fft.rfft(x, dim=1)
        a = torch.abs(s[:, 1:SAMPLE_LENGTH // 2 + 1])
        nf = a.median(dim=1, keepdim=True).values
        a = torch.log1p(a / (nf + 1e-6))
        a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-6)
        sq = self.c(a.unsqueeze(1)).transpose(1, 2)
        return sq, self.n(self.t(sq)).mean(dim=1)


class DG(nn.Module):
    def __init__(self, hd=HD):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(hd * 2, hd), nn.GELU(),
                                nn.Linear(hd, hd), nn.Sigmoid())
        self.f = nn.Sequential(nn.Linear(hd * 2, hd), nn.LayerNorm(hd),
                                nn.GELU(), nn.Dropout(0.1))
    def forward(self, ft, ff):
        g = self.g(torch.cat([ft, ff], dim=1))
        return self.f(torch.cat([g * ft + (1 - g) * ff, (ft + ff) * 0.5], dim=1))


class DX_PINN_V5(nn.Module):
    def __init__(self):
        super().__init__(); self.denoiser = Denoiser()
        self.st = Stem(); self.sf = Stem()
        self.et = TExt(); self.ef = FExt()
        self.dg = DG(); self.ccb = CausalConceptBottleneck(hidden_dim=HD)
        self.preprocessor = nn.Identity()
        self.time_encoder = None; self.freq_encoder = None
        self.cross_attention_fusion = None
        self.time_domain_head = None; self.frequency_domain_head = None
    def forward(self, x, t=None):
        xd, ne, es = self.denoiser(x)
        st = self.st(xd); sf = self.sf(xd)
        _, ft = self.et(st); _, ff = self.ef(xd)
        fused = self.dg(ft, ff); ccb = self.ccb(fused)
        B = x.shape[0]; z = torch.zeros(B, SAMPLE_LENGTH, device=x.device)
        return {"x_denoised": xd, "noise_est": ne, "est_snr": es,
                "cls_pred": ccb["cls_logits"], "c_hat": ccb["c_hat"],
                "domain_logits": ccb["domain_logits"],
                "z_causal": ccb["z_causal"], "z_context": ccb["z_context"],
                "shared_feature": fused, "time_feature": ft, "freq_feature": ff,
                "preprocessed_signal": xd, "raw_signal": x,
                "x_pred": z, "a_pred": z,
                "pred_amplitude": torch.zeros(B, SAMPLE_LENGTH // 2, device=x.device),
                "v_pred": z, "m": torch.zeros(B, 1, device=x.device),
                "c": torch.zeros(B, 1, device=x.device),
                "k": torch.zeros(B, 1, device=x.device)}
