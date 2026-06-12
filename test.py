"""DX-PINN 测试: 5 个实验"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset, get_dg_dataloader
from model.dx_pinn import DX_PINN
from model.ccb import compute_physics_concepts
from loss.concept_loss import cac_score
from utils import calculate_metrics
from xai import plot_concept_radar, plot_concept_vs_snr, plot_concept_tsne

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(path=None):
    if path is None:
        path = os.path.join(SAVE_PATH, "dx_pinn_final.pth")
    model = DX_PINN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def evaluate_with_snr(model, test_data, test_label, snr_db):
    dataset = BearingDataset(test_data, test_label, snr_db=snr_db, add_noise=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch_data, batch_label, batch_t in loader:
            batch_data = batch_data.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            batch_t = batch_t.to(DEVICE).requires_grad_(True)
            output = model(batch_data, batch_t)
            yt.extend(batch_label.cpu().numpy())
            yp.extend(torch.argmax(output["cls_pred"], dim=1).cpu().numpy())
    return calculate_metrics(yt, yp)


# ==================== 实验 1: 噪声鲁棒性 ====================
def experiment1_noise_robustness(model, n_runs=5):
    print("=" * 50 + "\n实验 1: 强噪声鲁棒性测试")
    train_data, _, test_data, _, _, test_label = load_dataset(TRAIN_WORK_CONDITION)
    results = {snr: [] for snr in SNR_LEVELS}
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        for snr in SNR_LEVELS:
            acc, _, _, _ = evaluate_with_snr(model, test_data, test_label, snr)
            results[snr].append(acc)
            print(f"  SNR={snr:3d}dB: Acc={acc:.4f}")
    rows = []
    for snr in SNR_LEVELS:
        accs = results[snr]
        rows.append({"SNR_dB": snr, "Acc_mean": np.mean(accs), "Acc_std": np.std(accs)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_PATH, "exp1_noise_robustness.csv"), index=False)
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["SNR_dB"], df["Acc_mean"], yerr=df["Acc_std"], marker='o', linewidth=2, capsize=4)
    plt.axvline(x=-10, color='red', linestyle='--', alpha=0.7, label='-10dB')
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90%')
    plt.xlabel("SNR (dB)"); plt.ylabel("准确率")
    plt.title("实验1: DX-PINN 噪声鲁棒性")
    plt.legend(); plt.grid(alpha=0.7); plt.ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, "exp1_noise_curve.png"), dpi=300); plt.close()
    return df


# ==================== 实验 2: 跨工况域泛化 ====================
def experiment2_cross_condition_dg(model, snr=-4):
    print("=" * 50 + "\n实验 2: 跨工况域泛化测试")
    conditions = [0, 1, 2, 3]
    results = np.full((4, 4), np.nan)
    for src in conditions:
        for tgt in conditions:
            if src == tgt:
                continue
            _, _, test_data, _, _, test_label = load_dataset(tgt)
            acc, _, _, _ = evaluate_with_snr(model, test_data, test_label, snr)
            results[src, tgt] = acc
            print(f"  {src}HP -> {tgt}HP @ {snr}dB: Acc={acc:.4f}")
    plt.figure(figsize=(8, 6))
    plt.imshow(results, cmap='YlOrRd', vmin=0.5, vmax=1.0, aspect='auto')
    for i in range(4):
        for j in range(4):
            if not np.isnan(results[i, j]):
                plt.text(j, i, f"{results[i, j]:.3f}", ha='center', va='center', fontsize=12)
    plt.xticks(range(4), [f"{c}HP" for c in conditions])
    plt.yticks(range(4), [f"{c}HP" for c in conditions])
    plt.xlabel("目标工况"); plt.ylabel("源工况")
    plt.title(f"实验2: 跨工况域泛化 (SNR={snr}dB)")
    plt.colorbar(label="准确率")
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, "exp2_dg_heatmap.png"), dpi=300); plt.close()
    valid = results[~np.isnan(results)]
    print(f"跨工况平均准确率: {valid.mean():.4f}")
    return results


# ==================== 实验 3: XAI 可解释性 ====================
def experiment3_xai(model):
    print("=" * 50 + "\n实验 3: XAI 事前可解释性评估")
    train_data, _, test_data, _, _, test_label = load_dataset(TRAIN_WORK_CONDITION)
    dataset = BearingDataset(test_data, test_label, snr_db=-4, add_noise=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_c_hat, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_data, batch_label, batch_t in tqdm(loader, desc="提取概念"):
            batch_data = batch_data.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            batch_t = batch_t.to(DEVICE).requires_grad_(True)
            output = model(batch_data, batch_t)
            all_c_hat.append(output["c_hat"].cpu())
            all_labels.append(batch_label.cpu())
    all_c_hat = torch.cat(all_c_hat, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    c_np = all_c_hat.numpy()
    l_np = all_labels.numpy()
    plot_concept_radar(c_np, l_np, os.path.join(SAVE_PATH, "exp3_concept_radar.png"))
    cac = cac_score(all_c_hat, all_labels, DEVICE)
    print(f"概念归因一致性 (CAC): {cac:.4f}")
    plot_concept_tsne(all_c_hat, all_labels, os.path.join(SAVE_PATH, "exp3_concept_tsne.png"))
    return cac


# ==================== 实验 4: 消融实验 (简化版) ====================
def experiment4_ablation():
    print("=" * 50 + "\n实验 4: 消融实验")
    train_data, _, test_data, _, _, test_label = load_dataset(TRAIN_WORK_CONDITION)
    snr_test = [-10, -6, 0]
    # 测试 Full DX-PINN 作为 baseline
    full_model = load_model()
    results = []
    for snr in snr_test:
        acc, _, _, _ = evaluate_with_snr(full_model, test_data, test_label, snr)
        results.append({"Variant": "Full DX-PINN", "SNR": snr, "Acc": acc})

    # 消融: w/o SUD (bypass denoiser)
    print("\n消融: w/o SUD")
    ablated = DX_PINN().to(DEVICE)
    state = torch.load(os.path.join(SAVE_PATH, "dx_pinn_final.pth"), map_location=DEVICE)
    ablated.load_state_dict(state)
    # Bypass denoiser with identity
    orig_forward = ablated.forward
    def no_sud_forward(self, x, t, labels=None, domain_ids=None, return_dg=False):
        x_proc = self.preprocessor(x)
        ts, tf = self.time_encoder(x_proc)
        fs, ff = self.freq_encoder(x)
        fused, _, _ = self.cross_attention_fusion(ts, fs)
        out = self.ccb(fused)
        return {"cls_pred": out["cls_logits"],
                "c_hat": out["c_hat"], "x_denoised": x,
                "x_pred": torch.zeros_like(x), "a_pred": torch.zeros_like(x),
                "pred_amplitude": torch.zeros(x.shape[0], SAMPLE_LENGTH // 2, device=x.device),
                "freq_pred_signal": torch.zeros_like(x),
                "v_pred": torch.zeros_like(x), "m": torch.zeros(x.shape[0], 1, device=x.device),
                "c": torch.zeros(x.shape[0], 1, device=x.device), "k": torch.zeros(x.shape[0], 1, device=x.device)}
    DX_PINN.forward = no_sud_forward
    for snr in snr_test:
        acc, _, _, _ = evaluate_with_snr(ablated, test_data, test_label, snr)
        results.append({"Variant": "w/o SUD", "SNR": snr, "Acc": acc})

    DX_PINN.forward = orig_forward  # restore
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_PATH, "exp4_ablation.csv"), index=False)
    return df


# ==================== 实验 5: 噪声类型泛化 ====================
def experiment5_noise_types(model):
    print("=" * 50 + "\n实验 5: 噪声类型泛化")
    train_data, _, test_data, _, _, test_label = load_dataset(TRAIN_WORK_CONDITION)
    results = {}
    noise_types = ["gaussian", "pink", "impulse", "mixed"]
    progress = tqdm(noise_types, desc="噪声类型")
    for ntype in progress:
        accs = []
        from data_loader import DGMultiConditionDataset as DG
        tmp = DG([TRAIN_WORK_CONDITION])
        for snr in SNR_LEVELS:
            dataset = BearingDataset(test_data, test_label, snr_db=snr, add_noise=False)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            model.eval()
            yt, yp = [], []
            with torch.no_grad():
                for batch_data, batch_label, batch_t in loader:
                    if ntype != "gaussian":
                        batch_data = torch.stack([
                            tmp.add_noise(batch_data[i], snr, ntype) for i in range(len(batch_data))
                        ])
                        batch_data = tmp.normalize_signal(batch_data)
                    else:
                        from utils import add_awgn_noise
                        batch_data = add_awgn_noise(batch_data, snr)
                        batch_data = (batch_data - batch_data.mean(dim=1, keepdim=True)) / (batch_data.std(dim=1, keepdim=True) + 1e-8)
                    batch_data = batch_data.to(DEVICE)
                    batch_label = batch_label.to(DEVICE)
                    batch_t = batch_t.to(DEVICE).requires_grad_(True)
                    output = model(batch_data, batch_t)
                    yt.extend(batch_label.cpu().numpy())
                    yp.extend(torch.argmax(output["cls_pred"], dim=1).cpu().numpy())
            acc, _, _, _ = calculate_metrics(yt, yp)
            accs.append(acc)
        results[ntype] = accs

    plt.figure(figsize=(10, 6))
    for ntype in noise_types:
        plt.plot(SNR_LEVELS, results[ntype], 'o-', linewidth=2, label=ntype)
    plt.axvline(x=-10, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)"); plt.ylabel("准确率")
    plt.title("实验5: 噪声类型泛化")
    plt.legend(); plt.grid(alpha=0.7); plt.ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, "exp5_noise_types.png"), dpi=300); plt.close()
    df = pd.DataFrame({"SNR_dB": SNR_LEVELS, **results})
    df.to_csv(os.path.join(SAVE_PATH, "exp5_noise_types.csv"), index=False)
    return df


def run_all_experiments():
    print("DX-PINN 完整实验矩阵")
    model = load_model()
    exp1 = experiment1_noise_robustness(model)
    exp2 = experiment2_cross_condition_dg(model, snr=-4)
    exp3 = experiment3_xai(model)
    exp4 = experiment4_ablation()
    exp5 = experiment5_noise_types(model)
    print("\n" + "=" * 50 + "\n所有实验完成!")
    print(f"Exp1 Acc@-10dB: {exp1[exp1['SNR_dB']==-10]['Acc_mean'].values[0]:.4f}")
    print(f"Exp3 CAC: {exp3:.4f}")


if __name__ == "__main__":
    run_all_experiments()
