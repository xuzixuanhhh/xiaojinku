"""XAI 可视化: 概念激活雷达图、SNR 变化曲线、t-SNE 概念分布"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from config import FAULT_CLASSES

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CN = ["冲击周期性", "频带能量集中度", "时域峭度",
      "包络调制深度", "频谱熵", "谐波衰减率"]
FN = list(FAULT_CLASSES.keys())


def plot_concept_radar(c_hat, labels, save_path, title="概念激活雷达图"):
    c_hat = np.nan_to_num(c_hat, nan=0.0, posinf=1.0, neginf=0.0)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        values = c_hat[mask].mean(axis=0).tolist()
        values += [values[0]]
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
        ax.plot(angles, values, 'o-', linewidth=2, label=FN[lbl], color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CN, fontsize=11)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_concept_vs_snr(all_snr_data, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, ax in enumerate(axes.flat):
        for fault_name, data in all_snr_data.items():
            snrs = data["snrs"]
            activations = [a[:, idx].mean() for a in data["c_hats"]]
            ax.plot(snrs, activations, 'o-', linewidth=2, markersize=4, label=fault_name)
        ax.set_title(CN[idx], fontsize=12)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("概念激活")
        ax.grid(True, linestyle="--", alpha=0.7)
    plt.suptitle("概念激活 vs SNR", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_concept_tsne(c_hat, labels, save_path, title="概念空间 t-SNE"):
    if torch.is_tensor(c_hat):
        c_np = c_hat.detach().cpu().numpy()
    else:
        c_np = c_hat
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    # NaN 防护
    c_np = np.nan_to_num(c_np, nan=0.0, posinf=1.0, neginf=0.0)
    if c_np.shape[0] > 5000:
        idx = np.random.choice(c_np.shape[0], 5000, replace=False)
        c_np = c_np[idx]
        labels = labels[idx]
    embedded = TSNE(n_components=2, random_state=42, perplexity=min(30, c_np.shape[0]-1), n_iter=1000).fit_transform(c_np)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(10):
        mask = labels == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   color=colors[i], label=FN[i], alpha=0.7, s=15)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
