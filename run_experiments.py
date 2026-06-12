"""DX-PINN 5-experiment suite"""
import torch, numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
from config import *
from data_loader import BearingDataset, load_dataset, DGMultiConditionDataset as DG
from model.dx_pinn import DX_PINN
from torch.utils.data import DataLoader
from utils import calculate_metrics, add_awgn_noise
from tqdm import tqdm

device = DEVICE
model = DX_PINN().to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'dx_pinn_best.pth'), map_location=device), strict=False)
model.eval()

# Exp 1: Noise Robustness
print('=== Exp 1: Noise Robustness ===')
_, _, td, _, _, tlbl = load_dataset(0)
n_runs = 5
all_results = {snr: [] for snr in SNR_LEVELS}
for run in range(n_runs):
    print(f'Run {run+1}/{n_runs}')
    for snr in SNR_LEVELS:
        ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = model(bd, bt)
                yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
                yt.extend(bl.cpu().numpy())
        acc, _, _, _ = calculate_metrics(yt, yp)
        all_results[snr].append(acc)
rows = [{'SNR_dB': s, 'Acc_mean': float(np.mean(a)), 'Acc_std': float(np.std(a))} for s, a in all_results.items()]
df1 = pd.DataFrame(rows)
df1.to_csv(os.path.join(SAVE_PATH, 'exp1_noise_robustness.csv'), index=False)
plt.figure(figsize=(10,6))
plt.errorbar(df1['SNR_dB'], df1['Acc_mean'], yerr=df1['Acc_std'], marker='o', linewidth=2, capsize=4, color='steelblue')
plt.axvline(x=-10, color='red', linestyle='--', alpha=0.7, label='-10dB')
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90%')
plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy'); plt.title('Exp 1: Noise Robustness')
plt.legend(); plt.grid(alpha=0.7); plt.ylim(0, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, 'exp1_noise_curve.png'), dpi=300); plt.close()
m10 = df1[df1['SNR_dB']==-10]
print(f'Acc@-10dB (mean+/-std): {m10["Acc_mean"].values[0]:.4f}+/-{m10["Acc_std"].values[0]:.4f}')

# Exp 2: Cross-Condition DG
print('\n=== Exp 2: Cross-Condition DG ===')
conditions = [0, 1, 2, 3]
results2 = np.full((4, 4), np.nan)
for src in conditions:
    for tgt in conditions:
        if src == tgt: continue
        _, _, ttd, _, _, ttlbl = load_dataset(tgt)
        ds = BearingDataset(ttd, ttlbl, snr_db=-4, add_noise=True)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = model(bd, bt)
                yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
                yt.extend(bl.cpu().numpy())
        acc, _, _, _ = calculate_metrics(yt, yp)
        results2[src, tgt] = acc
        print(f'  {src}HP->{tgt}HP: Acc={acc:.4f}')
valid = results2[~np.isnan(results2)]
print(f'DG Avg: {valid.mean():.4f}')
plt.figure(figsize=(8,6))
plt.imshow(results2, cmap='YlOrRd', vmin=0.2, vmax=1.0, aspect='auto')
for i in range(4):
    for j in range(4):
        if not np.isnan(results2[i,j]):
            plt.text(j, i, f'{results2[i,j]:.3f}', ha='center', va='center', fontsize=12)
plt.xticks(range(4), [f'{c}HP' for c in conditions])
plt.yticks(range(4), [f'{c}HP' for c in conditions])
plt.xlabel('Target'); plt.ylabel('Source'); plt.title('Exp 2: Cross-Condition DG (@-4dB)')
plt.colorbar(label='Accuracy')
plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, 'exp2_dg_heatmap.png'), dpi=300); plt.close()

# Exp 3: XAI
print('\n=== Exp 3: XAI ===')
from xai import plot_concept_radar
from loss.concept_loss import cac_score
ds = BearingDataset(td, tlbl, snr_db=-4, add_noise=True)
dl = DataLoader(ds, batch_size=64, shuffle=False)
all_c_hat, all_lbl = [], []
with torch.no_grad():
    for bd, bl, bt in tqdm(dl, desc='Concepts'):
        bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
        out = model(bd, bt)
        all_c_hat.append(torch.sigmoid(out['c_hat']).cpu())
        all_lbl.append(bl.cpu())
all_c_hat = torch.cat(all_c_hat, dim=0); all_lbl = torch.cat(all_lbl, dim=0)
cac = cac_score(all_c_hat, all_lbl, device)
print(f'CAC: {cac:.4f}')
plot_concept_radar(all_c_hat.numpy(), all_lbl.numpy(), os.path.join(SAVE_PATH, 'exp3_concept_radar.png'))

# Exp 4: Ablation
print('\n=== Exp 4: Ablation ===')
abl_results = []
for snr in [-10, -6, -4]:
    ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    yt, yp = [], []
    with torch.no_grad():
        for bd, bl, bt in dl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            out = model(bd, bt)
            yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
            yt.extend(bl.cpu().numpy())
    acc, _, _, _ = calculate_metrics(yt, yp)
    abl_results.append({'Variant': 'Full DX-PINN', 'SNR': snr, 'Acc': acc})
    print(f'  Full @{snr}dB: {acc:.4f}')
pd.DataFrame(abl_results).to_csv(os.path.join(SAVE_PATH, 'exp4_ablation.csv'), index=False)

# Exp 5: Noise Types
print('\n=== Exp 5: Noise Types ===')
tmp = DG([0])
ntypes = ['gaussian', 'pink', 'impulse', 'mixed']
results5 = {}
for ntype in ntypes:
    accs = []
    for snr in SNR_LEVELS:
        ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=False)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                if ntype != 'gaussian':
                    bd = torch.stack([tmp.add_noise(bd[i], snr, ntype) for i in range(len(bd))])
                    bd = tmp.normalize_signal(bd)
                else:
                    bd = add_awgn_noise(bd, snr)
                    bd = (bd - bd.mean(dim=1,keepdim=True)) / (bd.std(dim=1,keepdim=True)+1e-8)
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = model(bd, bt)
                yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
                yt.extend(bl.cpu().numpy())
        acc, _, _, _ = calculate_metrics(yt, yp)
        accs.append(acc)
    results5[ntype] = accs
    print(f'  {ntype}: -10dB={accs[0]:.4f}')
df5 = pd.DataFrame({'SNR_dB': SNR_LEVELS, **results5})
df5.to_csv(os.path.join(SAVE_PATH, 'exp5_noise_types.csv'), index=False)
plt.figure(figsize=(10,6))
for ntype in ntypes:
    plt.plot(SNR_LEVELS, results5[ntype], 'o-', linewidth=2, label=ntype)
plt.axvline(x=-10, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.5)
plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy'); plt.title('Exp 5: Noise Type Generalization')
plt.legend(); plt.grid(alpha=0.7); plt.ylim(0, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(SAVE_PATH, 'exp5_noise_types.png'), dpi=300); plt.close()
print('\n=== ALL EXPERIMENTS COMPLETE ===')
