import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *

# ===================== 1. 加性高斯白噪声添加函数 =====================
def add_awgn_noise(signal, snr_db):
    """
    向信号中添加指定信噪比的高斯白噪声
    :param signal: 输入信号，shape [batch_size, sample_length]
    :param snr_db: 信噪比，单位dB
    :return: 加噪后的信号
    """
    signal_power = torch.sum(signal ** 2, dim=1, keepdim=True) / signal.shape[1]
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    noisy_signal = signal + noise
    return noisy_signal

# ===================== 2. 可微希尔伯特变换（PyTorch实现）=====================
def hilbert_transform(x):
    """
    基于FFT实现的可微希尔伯特变换
    :param x: 输入信号，shape [batch_size, sample_length]
    :return: 解析信号，shape [batch_size, sample_length]
    """
    N = x.shape[1]
    X = torch.fft.fft(x, dim=1)
    # 构建希尔伯特变换的频域滤波器
    h = torch.zeros(N, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    h = h.unsqueeze(0)
    # 频域相乘后逆FFT
    x_hilbert = torch.fft.ifft(X * h, dim=1)
    return x_hilbert

# ===================== 3. 可微包络谱计算 =====================
def envelope_spectrum(x, fs=BEARING_PARAMS["fs"]):
    """
    计算信号的包络谱
    :param x: 输入信号，shape [batch_size, sample_length]
    :param fs: 采样频率
    :return: 包络谱幅值，频率轴
    """
    # 希尔伯特变换得到解析信号
    x_hilbert = hilbert_transform(x)
    # 包络信号
    envelope = torch.abs(x_hilbert)
    # 去直流分量
    envelope = envelope - torch.mean(envelope, dim=1, keepdim=True)
    # FFT计算包络谱
    N = envelope.shape[1]
    envelope_fft = torch.fft.fft(envelope, dim=1)
    envelope_spectrum_amp = torch.abs(envelope_fft)[:, :N//2] / N * 2
    # 频率轴
    freq_axis = torch.fft.fftfreq(N, 1/fs)[:N//2]
    return envelope_spectrum_amp, freq_axis.to(x.device)

# ===================== 4. 可微谱峭度计算 =====================
def spectral_kurtosis(x, fs=BEARING_PARAMS["fs"], nperseg=256, noverlap=128):
    """
    基于STFT计算信号的谱峭度，可微
    :param x: 输入信号，shape [batch_size, sample_length]
    :param fs: 采样频率
    :param nperseg: STFT窗口长度
    :param noverlap: 窗口重叠长度
    :return: 谱峭度曲线，频率轴
    """
    # 计算STFT
    window = torch.hann_window(nperseg, device=x.device)
    stft_result = torch.stft(
        x, n_fft=nperseg, hop_length=nperseg-noverlap, win_length=nperseg,
        window=window, return_complex=True
    )
    # 计算STFT幅值的平方
    stft_amp_sq = torch.abs(stft_result) ** 2
    # 计算四阶矩和二阶矩
    E4 = torch.mean(stft_amp_sq ** 2, dim=-1)
    E2 = torch.mean(stft_amp_sq, dim=-1) ** 2
    # 谱峭度计算公式
    sk = E4 / (E2 + 1e-10) - 2
    # 频率轴
    freq_axis = torch.fft.rfftfreq(nperseg, 1/fs)
    return sk, freq_axis.to(x.device)

# ===================== 5. 故障特征频率计算 =====================
def calculate_fault_frequency(work_condition):
    """
    计算指定工况下的各故障类型特征频率
    :param work_condition: 工况编号
    :return: 故障特征频率字典
    """
    n = SHAFT_SPEED[work_condition]
    fr = n / 60  # 轴转频，Hz
    D = BEARING_PARAMS["D"]
    d = BEARING_PARAMS["d"]
    Z = BEARING_PARAMS["Z"]
    alpha = BEARING_PARAMS["alpha"]
    
    # 计算各故障特征频率
    bpfo = Z / 2 * fr * (1 - d/D * np.cos(alpha))
    bpfi = Z / 2 * fr * (1 + d/D * np.cos(alpha))
    bsf = D / (2*d) * fr * (1 - (d/D * np.cos(alpha)) ** 2)
    
    return {
        "fr": fr,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf
    }

# ===================== 6. 分类指标计算 =====================
def calculate_metrics(y_true, y_pred):
    """
    计算分类指标：准确率、精确率、召回率、F1值
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, pre, rec, f1