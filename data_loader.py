import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from config import *

# ===================== 核心功能：读取单个.mat文件的驱动端振动数据 =====================
def load_mat_data(mat_path):
    """
    读取CWRU的.mat文件，提取驱动端振动数据
    适配CWRU官方mat文件格式，自动识别DE_time结尾的变量
    """
    try:
        mat_data = loadmat(mat_path)
        # 自动提取驱动端数据变量（以DE_time结尾）
        de_keys = [key for key in mat_data.keys() if key.endswith("DE_time")]
        if len(de_keys) == 0:
            raise ValueError(f"文件{mat_path}中未找到驱动端数据")
        vibration_data = mat_data[de_keys[0]].reshape(-1)  # 展平为一维数组
        return vibration_data
    except Exception as e:
        print(f"读取文件{mat_path}失败：{e}")
        # 兼容v7.3版本mat文件（可选）
        try:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                de_keys = [key for key in f.keys() if key.endswith("DE_time")]
                if len(de_keys) == 0:
                    raise ValueError(f"文件{mat_path}中未找到驱动端数据")
                vibration_data = np.array(f[de_keys[0]]).reshape(-1)
            return vibration_data
        except:
            raise ValueError(f"无法读取文件{mat_path}，请检查文件格式")

# ===================== 滑动窗口采样 =====================
def sliding_window_sampling(data, label, sample_length, sliding_step):
    """
    滑动窗口采样生成样本和标签
    :param data: 一维振动数据
    :param label: 故障类别标签
    :param sample_length: 样本长度
    :param sliding_step: 滑动步长
    :return: samples (N, sample_length), labels (N,)
    """
    samples = []
    labels = []
    data_len = len(data)
    # 滑动窗口遍历
    start_idx = 0
    while start_idx + sample_length <= data_len:
        sample = data[start_idx:start_idx+sample_length]
        samples.append(sample)
        labels.append(label)
        start_idx += sliding_step
    return np.array(samples), np.array(labels)

# ===================== 数据集加载主函数 =====================
def load_dataset(work_condition):
    """
    加载指定工况的完整数据集
    :param work_condition: 工况编号 0/1/2/3
    :return: train_data, val_data, test_data, train_label, val_label, test_label
    """
    all_samples = []
    all_labels = []
    work_condition_dir = os.path.join(DATA_ROOT, f"{work_condition}HP")
    
    # 检查工况文件夹是否存在
    if not os.path.exists(work_condition_dir):
        raise FileNotFoundError(f"工况文件夹不存在：{work_condition_dir}")
    
    # 遍历所有故障类别
    for fault_name, label in FAULT_CLASSES.items():
        fault_dir = os.path.join(work_condition_dir, fault_name)
        if not os.path.exists(fault_dir):
            raise FileNotFoundError(f"故障文件夹不存在：{fault_dir}")
        
        # 读取文件夹内的.mat文件
        mat_files = [f for f in os.listdir(fault_dir) if f.endswith(".mat")]
        if len(mat_files) == 0:
            raise FileNotFoundError(f"文件夹{fault_dir}中未找到.mat文件")
        mat_path = os.path.join(fault_dir, mat_files[0])
        
        # 读取振动数据
        vibration_data = load_mat_data(mat_path)
        # 归一化到[-1,1]
        vibration_data = 2 * (vibration_data - np.min(vibration_data)) / (np.max(vibration_data) - np.min(vibration_data)) - 1
        # 滑动窗口采样
        samples, labels = sliding_window_sampling(
            vibration_data, label, SAMPLE_LENGTH, SLIDING_STEP
        )
        
        all_samples.append(samples)
        all_labels.append(labels)
        print(f"工况{work_condition}HP-故障{fault_name}：生成{len(samples)}个样本")
    
    # 拼接所有样本
    all_samples = np.concatenate(all_samples, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 打乱数据集（固定随机种子保证可复现）
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(all_samples))
    all_samples = all_samples[shuffle_idx]
    all_labels = all_labels[shuffle_idx]
    
    # 划分训练集、验证集、测试集
    total_num = len(all_samples)
    train_num = int(total_num * TRAIN_RATIO)
    val_num = int(total_num * VAL_RATIO)
    
    train_data = all_samples[:train_num]
    train_label = all_labels[:train_num]
    val_data = all_samples[train_num:train_num+val_num]
    val_label = all_labels[train_num:train_num+val_num]
    test_data = all_samples[train_num+val_num:]
    test_label = all_labels[train_num+val_num:]
    
    print(f"\n工况{work_condition}HP数据集加载完成：")
    print(f"训练集：{len(train_data)}个样本，验证集：{len(val_data)}个样本，测试集：{len(test_data)}个样本")
    
    return train_data, val_data, test_data, train_label, val_label, test_label

# ===================== 自定义Dataset类 =====================
class BearingDataset(Dataset):
    def __init__(self, data, labels, snr_db=None, add_noise=True):
        """
        轴承数据集
        :param data: 原始数据
        :param labels: 标签
        :param snr_db: 信噪比（dB），如果为None则在[snr_min, snr_max]范围内随机
        :param add_noise: 是否添加噪声
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.t = torch.linspace(0, 1, SAMPLE_LENGTH, dtype=torch.float32)
        self.snr_db = snr_db
        self.add_noise = add_noise

    def __len__(self):
        return len(self.data)

    def add_gaussian_noise(self, signal, snr_db):
        """添加高斯白噪声"""
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise

    def normalize_signal(self, signal):
        return (signal - torch.mean(signal)) / (torch.std(signal) + 1e-8)

    def __getitem__(self, idx):
        signal = self.data[idx].clone()

        # 添加噪声
        if self.add_noise:
            if self.snr_db is not None:
                snr = self.snr_db
            else:
                # 随机SNR：-10dB到20dB
                snr = np.random.uniform(-10, 20)

            signal = self.add_gaussian_noise(signal, snr)

        signal = self.normalize_signal(signal)
        return signal, self.labels[idx], self.t


# ===================== 带噪声训练的数据集 =====================
class NoisyBearingDataset(Dataset):
    """带噪声的数据集，支持固定SNR或随机SNR"""
    def __init__(self, data, labels, snr_range=(-10, 20)):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.t = torch.linspace(0, 1, SAMPLE_LENGTH, dtype=torch.float32)
        self.snr_range = snr_range

    def __len__(self):
        return len(self.data)

    def add_gaussian_noise(self, signal, snr_db):
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise

    def normalize_signal(self, signal):
        return (signal - torch.mean(signal)) / (torch.std(signal) + 1e-8)

    def __getitem__(self, idx):
        signal = self.data[idx].clone()
        # 随机SNR
        snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        signal = self.add_gaussian_noise(signal, snr)
        signal = self.normalize_signal(signal)
        return signal, self.labels[idx], self.t

# ===================== 生成DataLoader =====================
def get_dataloader(work_condition, train_snr_range=(-10, 20), train_noise=False, val_noise=False, test_noise=True):
    """
    获取指定工况的训练、验证、测试DataLoader
    :param work_condition: 工况编号
    :param train_snr_range: 训练时随机SNR范围，如(-10, 20)
    :param train_noise: 训练集是否添加噪声
    :param val_noise: 验证集是否添加噪声（默认True）
    :param test_noise: 测试集是否添加噪声
    """
    train_data, val_data, test_data, train_label, val_label, test_label = load_dataset(work_condition)

    # 训练集：纯净数据或带随机噪声
    if train_noise:
        train_dataset = NoisyBearingDataset(train_data, train_label, snr_range=train_snr_range)
    else:
        train_dataset = BearingDataset(train_data, train_label, add_noise=False)

    # 验证集：带随机噪声（用于验证噪声鲁棒性）
    if val_noise:
        val_dataset = NoisyBearingDataset(val_data, val_label, snr_range=train_snr_range)
    else:
        val_dataset = BearingDataset(val_data, val_label, add_noise=False)

    # 测试集：不加噪声（用于评估）
    if test_noise:
        test_dataset = NoisyBearingDataset(test_data, test_label, snr_range=train_snr_range)
    else:
        test_dataset = BearingDataset(test_data, test_label, add_noise=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


class DGMultiConditionDataset(Dataset):
    """多工况域泛化数据集: (signal, label, t, domain_id)"""

    def __init__(self, conditions, snr_range=(-15, 5)):
        self.samples = []
        self.labels = []
        self.domain_ids = []
        self.t = torch.linspace(0, 1, SAMPLE_LENGTH, dtype=torch.float32)
        self.snr_range = snr_range
        for domain_id, wc in enumerate(conditions):
            data, _, _, lbl, _, _ = load_dataset(wc)
            data_t = torch.tensor(data, dtype=torch.float32)
            lbl_t = torch.tensor(lbl, dtype=torch.long)
            self.samples.append(data_t)
            self.labels.append(lbl_t)
            self.domain_ids.append(torch.full((len(lbl_t),), domain_id, dtype=torch.long))
        self.samples = torch.cat(self.samples, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.domain_ids = torch.cat(self.domain_ids, dim=0)

    def __len__(self):
        return len(self.samples)

    def add_noise(self, signal, snr_db, noise_type="gaussian"):
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        if noise_type == "gaussian":
            noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        elif noise_type == "pink":
            noise_white = torch.randn_like(signal)
            noise_fft = torch.fft.rfft(noise_white)
            freqs = torch.fft.rfftfreq(len(signal), 1.0)
            freqs[0] = 1.0
            noise_fft = noise_fft / torch.sqrt(freqs)
            noise = torch.fft.irfft(noise_fft, n=len(signal))
            noise = noise * torch.sqrt(noise_power / (noise.var() + 1e-8))
        elif noise_type == "impulse":
            noise = torch.zeros_like(signal)
            mask = torch.rand_like(signal) < 0.01
            noise[mask] = torch.randn(mask.sum()) * torch.sqrt(noise_power / 0.01)
        elif noise_type == "mixed":
            noise_g = torch.randn_like(signal) * torch.sqrt(noise_power * 0.5)
            noise_i = torch.zeros_like(signal)
            mask = torch.rand_like(signal) < 0.01
            noise_i[mask] = torch.randn(mask.sum()) * torch.sqrt(noise_power * 0.5 / 0.01)
            noise = noise_g + noise_i
        else:
            noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise

    def normalize_signal(self, signal):
        return (signal - torch.mean(signal)) / (torch.std(signal) + 1e-8)

    def __getitem__(self, idx):
        signal = self.samples[idx].clone()
        snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        signal = self.add_noise(signal, snr, "gaussian")
        signal = self.normalize_signal(signal)
        return signal, self.labels[idx], self.t, self.domain_ids[idx]


def get_dg_dataloader(conditions, snr_range=(-15, 5), batch_size=BATCH_SIZE):
    dataset = DGMultiConditionDataset(conditions, snr_range)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, dataset


# 测试代码（单独运行此文件可验证数据加载是否正常）
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader(TRAIN_WORK_CONDITION)
    for batch_data, batch_label, batch_t in train_loader:
        print(f"批次数据形状：{batch_data.shape}，批次标签形状：{batch_label.shape}，时间序列形状：{batch_t.shape}")
        break