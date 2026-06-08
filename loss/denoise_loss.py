"""去噪损失: 信号重建 + SNR 估计正则"""
import torch
import torch.nn as nn


def denoise_loss(denoised, target_clean, noise_est, est_snr, true_snr):
    """L_denoise = MSE(denoised, clean) + SNR_est_error + sparsity"""
    eps = 1e-6
    d_norm = (denoised - denoised.mean(dim=1, keepdim=True)) / (denoised.std(dim=1, keepdim=True) + eps)
    c_norm = (target_clean - target_clean.mean(dim=1, keepdim=True)) / (target_clean.std(dim=1, keepdim=True) + eps)
    L_recon = torch.mean((d_norm - c_norm) ** 2)
    L_snr = nn.MSELoss()(est_snr, true_snr.view(-1, 1))
    L_sparsity = torch.mean(torch.abs(noise_est))
    L_total = L_recon + 0.1 * L_snr + 0.05 * L_sparsity
    return L_total, {"L_denoise_recon": L_recon.item(), "L_denoise_snr": L_snr.item(), "L_denoise_sparsity": L_sparsity.item()}
