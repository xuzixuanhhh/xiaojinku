"""DX-PINN v6: Multi-Band Wavelet-inspired + ECA+SK backbone"""
import torch, torch.nn as nn, torch.nn.functional as F
from config import SAMPLE_LENGTH
from model.ccb import CausalConceptBottleneck
from model._base import ConvBNAct

HD = 96; DP = 0.15; NB = 4


class ECA(nn.Module):
    def __init__(self, c, k=5):
        super().__init__(); self.g = nn.AdaptiveAvgPool1d(1)
        self.c = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        y = self.g(x).squeeze(-1).unsqueeze(1)
        return x * torch.sigmoid(self.c(y).squeeze(1).unsqueeze(-1))


class SK(nn.Module):
    def __init__(self, c, ks=[3, 7, 15]):
        super().__init__()
        self.b = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False),
                nn.Conv1d(c, c, 1, bias=False), nn.BatchNorm1d(c), nn.GELU(),
            ) for k in ks])
        self.a = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                               nn.Linear(c, c // 4), nn.ReLU(),
                               nn.Linear(c // 4, len(ks)), nn.Softmax(dim=-1))
    def forward(self, x):
        o = torch.stack([b(x) for b in self.b], dim=2)
        w = self.a(x).unsqueeze(1).unsqueeze(-1)
        return (o * w).sum(dim=2) + x


class MultiBandDecomp(nn.Module):
    def __init__(self, nb=NB):
        super().__init__()
        self.bands = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, HD, kernel_size=2 ** (i + 1), stride=2 ** i,
                          padding=2 ** i, bias=False),
                nn.BatchNorm1d(HD), nn.GELU(),
            ) for i in range(nb)
        ])
        self.fw = nn.Parameter(torch.ones(nb) / nb)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        feats = []
        for i, band in enumerate(self.bands):
            bf = band(x)
            if bf.shape[2] != SAMPLE_LENGTH:
                bf = F.interpolate(bf, size=SAMPLE_LENGTH, mode='linear', align_corners=False)
            feats.append(bf)
        w = F.softmax(self.fw, dim=0)
        return sum(w[i] * feats[i] for i in range(NB))


class Proc(nn.Module):
    def __init__(self, c=HD):
        super().__init__()
        self.s = nn.Sequential(
            ConvBNAct(c, c, 7, stride=2, dropout=DP), SK(c), ECA(c),
            ConvBNAct(c, c, 5, stride=2, dropout=DP), SK(c), ECA(c),
        )
        self.p = nn.Conv1d(c, c, 1)
    def forward(self, x): return self.p(self.s(x))


class DX_PINN_V6(nn.Module):
    def __init__(self):
        super().__init__()
        self.decomp = MultiBandDecomp(); self.proc = Proc(HD); self.ec = ECA(HD)
        self.gs = nn.Sequential(
            ConvBNAct(1, HD // 4, 15, stride=2, dropout=DP),
            ConvBNAct(HD // 4, HD // 2, 9, stride=2, dropout=DP),
            ConvBNAct(HD // 2, HD, 7, stride=2, dropout=DP))
        self.sk = SK(HD); self.ge = ECA(HD)
        self.fus = nn.Sequential(nn.Linear(HD * 2, HD), nn.LayerNorm(HD),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(HD, HD))
        self.ccb = CausalConceptBottleneck(hidden_dim=HD)
        self.denoiser = nn.Identity(); self.preprocessor = nn.Identity()
        self.time_encoder = None; self.freq_encoder = None
        self.cross_attention_fusion = None
        self.time_domain_head = None; self.frequency_domain_head = None

    def forward(self, x, t=None):
        bf = self.decomp(x); bo = self.proc(bf); bv = self.ec(bo).mean(dim=2)
        g = self.gs(x.unsqueeze(1)); gv = self.ge(self.sk(g)).mean(dim=2)
        fused = self.fus(torch.cat([bv, gv], dim=1)); ccb = self.ccb(fused)
        B = x.shape[0]; z = torch.zeros(B, SAMPLE_LENGTH, device=x.device)
        return {"x_denoised": z, "noise_est": torch.zeros(B, 1, device=x.device),
                "est_snr": torch.zeros(B, 1, device=x.device),
                "cls_pred": ccb["cls_logits"], "c_hat": ccb["c_hat"],
                "domain_logits": ccb["domain_logits"],
                "z_causal": ccb["z_causal"], "z_context": ccb["z_context"],
                "shared_feature": fused, "time_feature": bv, "freq_feature": gv,
                "preprocessed_signal": x, "raw_signal": x,
                "x_pred": z, "a_pred": z,
                "pred_amplitude": torch.zeros(B, SAMPLE_LENGTH // 2, device=x.device),
                "v_pred": z, "m": torch.zeros(B, 1, device=x.device),
                "c": torch.zeros(B, 1, device=x.device),
                "k": torch.zeros(B, 1, device=x.device)}
