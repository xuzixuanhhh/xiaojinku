"""DX-PINN: Deep eXplainable Physics-Informed Network"""
import torch
import torch.nn as nn
from model.sud import StockwellUnrolledDenoiser
from model.ccb import CausalConceptBottleneck
from model.mvm_cdg import MVMCDG

from model._base import (
    RobustSignalPreprocessor, TimeDomainEncoder, FrequencyDomainEncoder,
    CrossAttentionFusion, TimeDomainHead, FrequencyDomainHead,
)
from config import SAMPLE_LENGTH


class DX_PINN(nn.Module):
    """DX-PINN v10: SUD + FFT Encoder + CCB + MVM-CDG (v6 architecture, optimized training)"""

    def __init__(self):
        super().__init__()
        self.denoiser = StockwellUnrolledDenoiser()
        self.preprocessor = RobustSignalPreprocessor()
        self.time_encoder = TimeDomainEncoder()
        self.freq_encoder = FrequencyDomainEncoder()
        self.cross_attention_fusion = CrossAttentionFusion()
        self.time_domain_head = TimeDomainHead()
        self.frequency_domain_head = FrequencyDomainHead()
        self.ccb = CausalConceptBottleneck()
        self.mvm_cdg = MVMCDG()

    def forward(self, x, t, labels=None, domain_ids=None, return_dg=False):
        x_raw = x
        x_denoised, noise_est, est_snr = self.denoiser(x)
        x_proc = self.preprocessor(x_denoised)
        time_seq, time_feature = self.time_encoder(x_proc)
        freq_seq, freq_feature = self.freq_encoder(x_denoised)
        fused_feature, time_feature, freq_feature = self.cross_attention_fusion(time_seq, freq_seq)

        ccb_output = self.ccb(fused_feature)
        c_hat = ccb_output["c_hat"]
        cls_pred = ccb_output["cls_logits"]

        x_pred, a_pred, pinn_output = self.time_domain_head(fused_feature, t)
        pred_amplitude = self.frequency_domain_head(fused_feature)
        spectrum_full = torch.complex(pred_amplitude, torch.zeros_like(pred_amplitude))
        freq_pred_signal = torch.fft.irfft(spectrum_full, n=SAMPLE_LENGTH, dim=1)

        output = {
            "x_denoised": x_denoised, "noise_est": noise_est, "est_snr": est_snr,
            "cls_pred": cls_pred, "c_hat": c_hat,
            "domain_logits": ccb_output["domain_logits"],
            "z_causal": ccb_output["z_causal"],
            "z_context": ccb_output["z_context"],
            "shared_feature": fused_feature,
            "time_feature": time_feature, "freq_feature": freq_feature,
            "preprocessed_signal": x_proc, "raw_signal": x_raw,
            "x_pred": x_pred, "a_pred": a_pred,
            "pred_amplitude": pred_amplitude,
            "freq_pred_signal": freq_pred_signal,
            "v_pred": pinn_output["v_pred"],
            "m": pinn_output["m"], "c": pinn_output["c"], "k": pinn_output["k"],
        }

        if return_dg and labels is not None and domain_ids is not None:
            dg_output = self.mvm_cdg(time_feature, freq_feature, c_hat, labels, domain_ids)
            output["dg_output"] = dg_output

        return output

    def set_grl_alpha(self, alpha):
        self.ccb.set_grl_alpha(alpha)
