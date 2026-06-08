"""DX-PINN-Lite: Simplified model for fast 100-epoch convergence
Reductions: ISTA 5->2, kernels 128->64, hidden_dim 128->64, encoder layers 3->2
"""
import torch
import torch.nn as nn
from model.sud import StockwellUnrolledDenoiser
from model.ccb import CausalConceptBottleneck
from model._base import (
    RobustSignalPreprocessor, TimeDomainEncoder, FrequencyDomainEncoder,
    CrossAttentionFusion, TimeDomainHead, FrequencyDomainHead,
)
from config import SAMPLE_LENGTH

HIDDEN_DIM = 64
ENC_LAYERS = 4  # TimeEncoder uses ENC_LAYERS//2 = 2
FREQ_LAYERS = 2
NUM_ISTA = 2
NUM_KERNELS = 64


class DX_PINN_Lite(nn.Module):
    """DX-PINN-Lite: Reduced complexity for fast 100-epoch training"""

    def __init__(self):
        super().__init__()
        self.denoiser = StockwellUnrolledDenoiser(
            num_kernels=NUM_KERNELS, kernel_size=15, num_ista_layers=NUM_ISTA)
        self.preprocessor = RobustSignalPreprocessor()
        self.time_encoder = TimeDomainEncoder(
            hidden_dim=HIDDEN_DIM, num_layers=ENC_LAYERS)
        self.freq_encoder = FrequencyDomainEncoder(
            hidden_dim=HIDDEN_DIM, num_layers=FREQ_LAYERS)
        self.cross_attention_fusion = CrossAttentionFusion(
            hidden_dim=HIDDEN_DIM)
        self.time_domain_head = TimeDomainHead(hidden_dim=HIDDEN_DIM)
        self.frequency_domain_head = FrequencyDomainHead(hidden_dim=HIDDEN_DIM)
        self.ccb = CausalConceptBottleneck(hidden_dim=HIDDEN_DIM)

    def forward(self, x, t):
        x_denoised, noise_est, est_snr = self.denoiser(x)
        x_proc = self.preprocessor(x_denoised)
        time_seq, time_feature = self.time_encoder(x_proc)
        freq_seq, freq_feature = self.freq_encoder(x_denoised)
        fused_feature, time_feature, freq_feature = self.cross_attention_fusion(
            time_seq, freq_seq)

        ccb_output = self.ccb(fused_feature)

        x_pred, a_pred, pinn_output = self.time_domain_head(fused_feature, t)
        pred_amplitude = self.frequency_domain_head(fused_feature)

        return {
            "x_denoised": x_denoised, "noise_est": noise_est, "est_snr": est_snr,
            "cls_pred": ccb_output["cls_logits"], "c_hat": ccb_output["c_hat"],
            "domain_logits": ccb_output["domain_logits"],
            "z_causal": ccb_output["z_causal"],
            "z_context": ccb_output["z_context"],
            "shared_feature": fused_feature,
            "time_feature": time_feature, "freq_feature": freq_feature,
            "preprocessed_signal": x_proc, "raw_signal": x,
            "x_pred": x_pred, "a_pred": a_pred,
            "pred_amplitude": pred_amplitude,
            "v_pred": pinn_output["v_pred"],
            "m": pinn_output["m"], "c": pinn_output["c"], "k": pinn_output["k"],
        }
