import torch
import torch.nn as nn
import numpy as np
from utils import envelope_spectrum
from config import *

# ===================== 1. 数据拟合损失 L_data =====================
def data_fitting_loss(a_pred, x_meas):
    """
    数据拟合损失，MSE损失
    :param a_pred: 预测加速度（已归一化），shape [batch_size, sample_length]
    :param x_meas: 实测振动信号（已归一化到[-1,1]），shape [batch_size, sample_length]
    :return: L_data
    """
    return nn.MSELoss()(a_pred, x_meas)

# ===================== 2. 振动信号重建损失 L_recon =====================
def signal_reconstruction_loss(x_pred, x_meas):
    """
    振动信号重建损失
    :param x_pred: 预测位移，shape [batch_size, sample_length]
    :param x_meas: 实测振动信号，shape [batch_size, sample_length]
    :return: L_recon
    """
    eps = 1e-6
    x_pred_norm = (x_pred - x_pred.mean(dim=1, keepdim=True)) / (x_pred.std(dim=1, keepdim=True) + eps)
    x_meas_norm = (x_meas - x_meas.mean(dim=1, keepdim=True)) / (x_meas.std(dim=1, keepdim=True) + eps)
    return torch.mean((x_pred_norm - x_meas_norm) ** 2)

# ===================== 3. 包络谱峰值匹配损失 L_spec =====================
def envelope_spectrum_loss(pred_signal, meas_signal, fault_label, work_condition=TRAIN_WORK_CONDITION, K=HARMONIC_NUM):
    """
    包络谱峰值匹配损失
    :param pred_signal: 频域分支预测信号，shape [batch_size, sample_length]
    :param meas_signal: 实测信号，shape [batch_size, sample_length]
    :param fault_label: 故障标签，shape [batch_size]
    :param work_condition: 工况编号
    :param K: 倍频数
    :return: L_spec
    """
    pred_spec, freq_axis = envelope_spectrum(pred_signal)
    meas_spec, _ = envelope_spectrum(meas_signal)
    
    from utils import calculate_fault_frequency
    fault_freq = calculate_fault_frequency(work_condition)
    batch_size = pred_signal.shape[0]
    L_spec = 0.0
    
    for i in range(batch_size):
        label = fault_label[i].item()
        if label == 0:
            ff = fault_freq["fr"]
        elif 1 <= label <= 3:
            ff = fault_freq["bpfi"]
        elif 4 <= label <= 6:
            ff = fault_freq["bpfo"]
        else:
            ff = fault_freq["bsf"]
        
        loss = 0.0
        for k in range(1, K+1):
            target_freq = k * ff
            freq_idx = torch.argmin(torch.abs(freq_axis - target_freq))
            loss += (pred_spec[i, freq_idx] - meas_spec[i, freq_idx]) ** 2
        L_spec += loss / K
    
    L_spec = L_spec / batch_size
    return L_spec

# ===================== 5. 故障特征频率匹配损失 L_energy =====================
def energy_constraint_loss(pred_signal, meas_signal, fault_label, work_condition=TRAIN_WORK_CONDITION):
    """
    故障特征频率匹配损失
    找到预测信号频谱中的峰值频率，让它接近对应故障类型的特征频率
    使用 Gaussian 形式：exp(-d^2)，越接近目标频率损失越小
    """
    fs = BEARING_PARAMS["fs"]
    N = pred_signal.shape[1]

    pred_fft = torch.fft.fft(pred_signal, dim=1)
    pred_amp = torch.abs(pred_fft)[:, 1:N//2]  # 去掉直流分量
    freq_axis = torch.fft.fftfreq(N, 1/fs)[1:N//2].to(pred_signal.device)

    from utils import calculate_fault_frequency
    fault_freq = calculate_fault_frequency(work_condition)
    batch_size = pred_signal.shape[0]
    L_energy = 0.0

    for i in range(batch_size):
        label = fault_label[i].item()

        # 根据标签确定目标频率集合（包含倍频）
        if label == 0:  # NC - 转频
            fr = fault_freq["fr"]
            target_freqs = [fr, 2*fr, 3*fr]
        elif 1 <= label <= 3:  # IF - 内圈故障
            bpfi = fault_freq["bpfi"]
            target_freqs = [bpfi, 2*bpfi, 3*bpfi, 4*bpfi]
        elif 4 <= label <= 6:  # OF - 外圈故障
            bpfo = fault_freq["bpfo"]
            target_freqs = [bpfo, 2*bpfo, 3*bpfo, 4*bpfo]
        else:  # BF - 滚动体故障
            bsf = fault_freq["bsf"]
            target_freqs = [bsf, 2*bsf, 3*bsf, 4*bsf]

        # 找到预测频谱的主峰频率
        peak_idx = torch.argmax(pred_amp[i])
        peak_freq = freq_axis[peak_idx]

        # 计算与每个目标频率的距离
        target_tensor = torch.tensor(target_freqs, device=pred_signal.device)
        distances = torch.abs(peak_freq - target_tensor)

        # Gaussian 惩罚：越接近目标频率，损失越小（趋近于0）
        # 使用 100Hz 作为 sigma，控制惩罚范围
        sigma = 100.0
        gaussian_penalty = torch.exp(-(distances ** 2) / (2 * sigma ** 2))

        # 取最大值（最接近的惩罚），然后取负（惩罚越小损失越小）
        # Gaussian 在 0 时为 1，所以用 1 - max 来得到损失
        loss = 1.0 - torch.max(gaussian_penalty)

        L_energy += loss

    L_energy = L_energy / batch_size
    return L_energy

# ===================== 6. 分类交叉熵损失 L_cls =====================
def classification_loss(cls_pred, label):
    """
    分类交叉熵损失
    :param cls_pred: 分类预测概率，shape [batch_size, num_classes]
    :param label: 真实标签，shape [batch_size]
    :return: L_cls
    """
    return nn.CrossEntropyLoss()(cls_pred, label)

# ===================== 7. PINN 物理约束损失 L_physics =====================
def pinn_physics_loss(model_output, meas_signal, fault_label, work_condition=TRAIN_WORK_CONDITION):
    """
    PINN 物理约束损失
    强制满足 MCK 二阶振动方程: m·x'' + c·x' + k·x = F(t)
    
    对于自由振动（F ≈ 0）或弱激励：
    m·x'' + c·x' + k·x ≈ 0
    
    :param model_output: 模型输出，包含 x_pred, v_pred, a_pred, m, c, k
    :param meas_signal: 实测振动信号
    :param fault_label: 故障标签
    :param work_condition: 工况编号
    :return: L_physics
    """
    x_pred = model_output["x_pred"]
    v_pred = model_output["v_pred"]
    a_pred = model_output["a_pred"]  # 来自 PINN 的物理推导加速度
    m = model_output["m"]
    c = model_output["c"]
    k = model_output["k"]
    
    # MCK 方程残差: m·a + c·v + k·x = 0 (自由振动)
    mck_residual = m * a_pred + c * v_pred + k * x_pred
    scale = (
        torch.abs(m * a_pred).mean(dim=1, keepdim=True)
        + torch.abs(c * v_pred).mean(dim=1, keepdim=True)
        + torch.abs(k * x_pred).mean(dim=1, keepdim=True)
        + 1e-6
    )
    mck_residual = mck_residual / scale

    # 残差应该接近于 0，归一化后避免刚度项主导总损失
    L_physics_mck = torch.mean(mck_residual ** 2)

    # 额外约束：加速度形态与实测信号一致，使用样本级归一化避免幅值主导
    a_pred_norm = (a_pred - a_pred.mean(dim=1, keepdim=True)) / (a_pred.std(dim=1, keepdim=True) + 1e-6)
    meas_norm = (meas_signal - meas_signal.mean(dim=1, keepdim=True)) / (meas_signal.std(dim=1, keepdim=True) + 1e-6)
    L_physics_data = torch.mean((a_pred_norm - meas_norm) ** 2)
    
    # 额外约束：速度应该与位移的微分一致（用于检验自动微分的正确性）
    # 直接从位移用梯度计算速度，然后与 v_pred 比较
    v_from_x = torch.gradient(x_pred, dim=1)[0]
    L_physics_gradient = torch.mean((v_pred - v_from_x) ** 2)
    
    # 综合物理损失
    L_physics = L_physics_mck + 0.5 * L_physics_data + 0.1 * L_physics_gradient
    
    return L_physics


# ===================== 8. MCK 参数正则化损失 L_mck_regularization =====================
def mck_parameter_loss(model_output, meas_signal):
    """
    MCK 参数正则化损失
    - 防止参数过大或过小
    - 鼓励参数在合理范围内
    
    :param model_output: 模型输出，包含 m, c, k
    :param meas_signal: 实测信号
    :return: L_mck_reg
    """
    m = model_output["m"]
    c = model_output["c"]
    k = model_output["k"]
    
    # 提取参数并限制范围
    # 使用 log 尺度的正则化，使参数在合理范围
    L_m = torch.mean((torch.log(m + 1e-8) - np.log(1.0)) ** 2)  # m ≈ 1.0
    L_c = torch.mean((torch.log(c + 1e-8) - np.log(5.0)) ** 2)  # c ≈ 5.0
    L_k = torch.mean((torch.log(k + 1e-8) - np.log(1000.0)) ** 2)  # k ≈ 1000.0
    
    # 归一化到合理范围
    L_mck_reg = L_m + L_c + L_k
    
    return L_mck_reg


# ===================== 9. 特征频率约束损失 L_fault_freq（改进版）=====================
def fault_frequency_loss(pred_amplitude, meas_signal, fault_label, work_condition=TRAIN_WORK_CONDITION, K=HARMONIC_NUM):
    """
    特征频率约束损失（改进版）
    核心思想：直接对实测信号计算FFT，约束FFT峰值接近故障特征频率

    :param pred_amplitude: 预测的幅度谱，shape [batch_size, num_freq_bins]
    :param meas_signal: 实测振动信号，shape [batch_size, sample_length]
    :param fault_label: 故障标签，shape [batch_size]
    :param work_condition: 工况编号
    :param K: 考虑的倍频数
    :return: L_fault_freq
    """
    fs = BEARING_PARAMS["fs"]
    N = meas_signal.shape[1]
    
    from utils import calculate_fault_frequency
    fault_freq = calculate_fault_frequency(work_condition)
    
    batch_size = meas_signal.shape[0]
    device = meas_signal.device
    
    # 对实测信号计算FFT
    fft_result = torch.fft.fft(meas_signal, dim=1)
    meas_amplitude = torch.abs(fft_result)[:, 1:N//2]  # 去掉直流分量
    freq_axis = torch.fft.fftfreq(N, 1/fs)[1:N//2].to(device)
    
    L_fault_freq = 0.0
    
    for i in range(batch_size):
        label = fault_label[i].item()
        
        # 根据标签确定目标频率集合
        if label == 0:
            fr = fault_freq["fr"]
            target_freqs = [fr * k for k in range(1, K + 1)]
        elif 1 <= label <= 3:
            bpfi = fault_freq["bpfi"]
            target_freqs = [bpfi * k for k in range(1, K + 1)]
        elif 4 <= label <= 6:
            bpfo = fault_freq["bpfo"]
            target_freqs = [bpfo * k for k in range(1, K + 1)]
        else:
            bsf = fault_freq["bsf"]
            target_freqs = [bsf * k for k in range(1, K + 1)]
        
        # 方法1：找实测FFT的峰值频率
        peak_idx = torch.argmax(meas_amplitude[i])
        peak_freq = freq_axis[peak_idx]
        
        # 方法2：用预测幅度谱加权计算期望频率（结合预测信息）
        pred_amp = pred_amplitude[i][:len(freq_axis)]
        weights = torch.nn.functional.softmax(pred_amp * 10.0, dim=0)
        expected_freq = torch.sum(weights * freq_axis)
        
        # 融合两种方法
        combined_freq = 0.6 * peak_freq + 0.4 * expected_freq
        
        # 计算与目标频率的距离
        target_tensor = torch.tensor(target_freqs, device=device)
        distances = torch.abs(combined_freq - target_tensor)
        min_distance = torch.min(distances)
        
        # 归一化损失
        L_fault_freq += min_distance / (fs / 2)
    
    return L_fault_freq / batch_size


# ===================== 总损失函数（改进版）=====================
def total_loss(model_output, meas_signal, t, fault_label, loss_weights=LOSS_WEIGHTS):
    """
    计算总损失函数（改进版）
    - 分类损失 L_cls（带标签平滑）
    - 重建损失 L_recon
    - 特征频率约束损失 L_fault_freq（直接约束实测FFT）
    - PINN 物理损失 L_physics
    - MCK 参数正则化损失 L_mck_reg
    """
    cls_pred = model_output["cls_pred"]
    x_pred = model_output["x_pred"]
    a_pred = model_output["a_pred"]
    pred_amplitude = model_output["pred_amplitude"]
    
    # 分类损失（标签平滑）
    L_cls = label_smoothing_loss(cls_pred, fault_label, smoothing=0.1)
    
    # 重建损失
    L_recon = signal_reconstruction_loss(x_pred, meas_signal)
    
    # 特征频率约束损失
    L_fault_freq = fault_frequency_loss(pred_amplitude, meas_signal, fault_label)
    
    # PINN 物理约束损失
    L_physics = pinn_physics_loss(model_output, meas_signal, fault_label)
    L_mck_reg = mck_parameter_loss(model_output, meas_signal)
    
    # 加权总损失
    L_total = (
        loss_weights["lambda_cls"] * L_cls
        + loss_weights["lambda_recon"] * L_recon
        + loss_weights["lambda_fault_freq"] * L_fault_freq
        + loss_weights["lambda_physics"] * L_physics
        + loss_weights["lambda_mck_reg"] * L_mck_reg
    )
    
    return L_total, {
        "L_total": L_total.item(),
        "L_cls": L_cls.item(),
        "L_recon": L_recon.item(),
        "L_fault_freq": L_fault_freq.item(),
        "L_physics": L_physics.item(),
        "L_mck_reg": L_mck_reg.item(),
    }


def label_smoothing_loss(pred, target, smoothing=0.1):
    """
    带标签平滑的交叉熵损失
    :param pred: 模型预测 logits
    :param target: 真实标签
    :param smoothing: 平滑因子
    """
    n_classes = pred.size(1)
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (n_classes - 1)
    
    one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    smooth_target = one_hot * confidence + (1 - one_hot) * smooth_value
    
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    loss = -(smooth_target * log_probs).sum(dim=1).mean()
    
    return loss
