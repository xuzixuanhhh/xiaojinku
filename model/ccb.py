"""Causal Concept Bottleneck — 因果概念瓶颈 + 域对抗解耦"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ENCODER_HIDDEN_DIM, NUM_CLASSES, BEARING_PARAMS, HARMONIC_NUM


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class CausalConceptBottleneck(nn.Module):
    """因果概念瓶颈 (软瓶颈模式)

    训练时: 分类器可以同时看到概念和 z_causal (梯度流)
    推理时: 分类器只看概念 (XAI 透明)

    输入: fused_feature [B, 128]
    输出:
        c_hat: 概念激活 [B, 6]
        cls_logits: 分类 logits [B, 10]
        domain_logits: 域分类 logits [B, 4]
    """

    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, num_concepts=6,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.causal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.concept_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_concepts),
        )  # linear output, no saturation
        # 软瓶颈: wider classifier (128→64→10 for clean data >98%)
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # 推理时用: 仅概念→分类
        self.inference_classifier = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.grl = GRL()
        self.soft_gate = 1.0  # 1.0=训练模式(软瓶颈), 0.0=推理模式(硬瓶颈)

    def set_soft_gate(self, gate):
        self.soft_gate = gate

    def forward(self, fused_feature):
        z_causal = F.relu(self.causal_proj(fused_feature))
        z_context = self.context_proj(fused_feature)
        c_hat = self.concept_net(z_causal)

        # 软瓶颈: concat(概念, z_causal) → 分类
        soft_input = torch.cat([c_hat, z_causal], dim=1)
        cls_logits = self.classifier(soft_input)

        z_context_rev = self.grl(z_context)
        domain_logits = self.domain_classifier(z_context_rev)

        return {
            "c_hat": c_hat,
            "cls_logits": cls_logits,
            "z_causal": z_causal,
            "z_context": z_context,
            "domain_logits": domain_logits,
        }

    def set_grl_alpha(self, alpha):
        self.grl.set_alpha(alpha)


# 期望归因先验 (每个故障类型的期望概念激活)
EAP_PRIORS = torch.tensor([
    [0.0, 0.1, 0.1, 0.0, 0.8, 0.0],  # NC
    [0.8, 0.7, 0.6, 0.7, 0.2, 0.5],  # IF1
    [0.8, 0.7, 0.6, 0.7, 0.2, 0.5],  # IF2
    [0.8, 0.7, 0.6, 0.7, 0.2, 0.5],  # IF3
    [0.9, 0.8, 0.7, 0.8, 0.2, 0.6],  # OF1
    [0.9, 0.8, 0.7, 0.8, 0.2, 0.6],  # OF2
    [0.9, 0.8, 0.7, 0.8, 0.2, 0.6],  # OF3
    [0.7, 0.5, 0.5, 0.5, 0.3, 0.3],  # BF1
    [0.7, 0.5, 0.5, 0.5, 0.3, 0.3],  # BF2
    [0.7, 0.5, 0.5, 0.5, 0.3, 0.3],  # BF3
], dtype=torch.float32)


def compute_physics_concepts(signals):
    """从信号计算物理概念值 (用于 L_concept 监督)

    返回: dict with c1-c6, each [B]
    """
    B, L = signals.shape
    device = signals.device
    fs = BEARING_PARAMS["fs"]

    eps = 1e-8

    # c3: 时域峭度
    m2 = ((signals - signals.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
    m4 = ((signals - signals.mean(dim=1, keepdim=True)) ** 4).mean(dim=1)
    c3 = m4 / (m2 ** 2 + eps)

    # c5: 频谱熵
    fft = torch.fft.rfft(signals, dim=1)
    amp = torch.abs(fft)[:, 1:]
    amp_norm = amp / (amp.sum(dim=1, keepdim=True) + eps)
    c5 = -(amp_norm * torch.log(amp_norm + eps)).sum(dim=1)

    # 包络谱
    from utils import hilbert_transform
    analytic = hilbert_transform(signals)
    envelope = torch.abs(analytic)
    env_fft = torch.fft.rfft(envelope, dim=1)
    env_amp = torch.abs(env_fft)[:, 1:]

    c1 = torch.zeros(B, device=device)
    c2 = torch.zeros(B, device=device)
    c4 = torch.zeros(B, device=device)
    c6 = torch.zeros(B, device=device)

    for i in range(B):
        peak_idx = torch.argmax(env_amp[i, :L // 4])
        peak_val = env_amp[i, peak_idx]
        nb = env_amp[i, max(0, peak_idx - 5): min(L // 2 - 1, peak_idx + 5)]
        c4[i] = peak_val / (nb.median() + eps)

        band_s = max(0, peak_idx - 10)
        band_e = min(L // 2 - 1, peak_idx + 10)
        band_energy = env_amp[i, band_s:band_e].sum()
        c2[i] = band_energy / (env_amp[i].sum() + eps)

        sig_i = signals[i:i + 1].unsqueeze(1)  # [1, 1, L]
        kernel = signals[i:i + 1].flip(-1).unsqueeze(1)  # [1, 1, L]
        acf = F.conv1d(sig_i, kernel, padding=L - 1).squeeze()[L:]
        peaks = torch.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0]
        if len(peaks) >= 2:
            intervals = peaks[1:] - peaks[:-1]
            c1[i] = 1.0 / (intervals.float().var() + eps)

        if peak_idx > 0 and peak_idx * 3 < L // 2:
            vals = [env_amp[i, peak_idx * k].item() for k in range(1, HARMONIC_NUM + 1) if peak_idx * k < L // 2 - 1]
            if len(vals) >= 2:
                vals_t = torch.tensor(vals, device=device)
                log_v = torch.log(vals_t + eps)
                x = torch.arange(len(log_v), dtype=torch.float32, device=device)
                n = len(log_v)
                slope = (n * (x * log_v).sum() - x.sum() * log_v.sum()) / (n * (x ** 2).sum() - x.sum() ** 2 + eps)
                c6[i] = torch.abs(slope)

    # 归一化 (避免 inplace)
    c1 = c1 / (c1.max() + eps)
    c2 = c2 / (c2.max() + eps)
    c3 = c3 / (c3.max() + eps)
    c4 = c4 / (c4.max() + eps)
    c5 = c5 / (c5.max() + eps)
    c6 = c6 / (c6.max() + eps)

    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6}
