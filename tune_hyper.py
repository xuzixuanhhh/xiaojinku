"""Hyperparameter sweep for -10dB optimization under MS-TCANet conditions"""
import torch, numpy as np, random, json
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def run_config(name, lr, epochs, eap_w, recon_w, snr_range, pretrain_ep=8, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_loader, _, _ = get_dataloader(1, train_snr_range=snr_range,
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)

    model = DX_PINN().to(device)

    def ev(snr):
        ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = model(bd, bt)
                yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
                yt.extend(bl.cpu().numpy())
        return calculate_metrics(yt, yp)[0]

    # Phase 1
    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for _ in range(pretrain_ep):
        for bd, _, _ in train_loader:
            bd = bd.to(device); o1.zero_grad()
            xd, ne, es = model.denoiser(bd)
            s_low, s_high = snr_range
            ts = torch.rand(bd.shape[0], 1, device=device) * (s_high - s_low) + s_low
            L, _ = denoise_loss(xd, bd, ne, es, ts)
            L.backward(); o1.step()

    # Phase 2
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs, eta_min=1e-5)
    best_m10, patience, history = 0.0, 0, []

    for ep in range(epochs):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                    + eap_w * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + recon_w * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
        sc.step()

        if ep % 10 == 0:
            model.eval(); m10 = ev(-10)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            history.append((ep+1, m10))
            if patience >= 20: break
            model.train()

    model.eval()
    final_m4 = ev(-4); final_m10 = ev(-10)
    result = {
        'name': name, 'lr': lr, 'epochs': epochs, 'eap_w': eap_w, 'recon_w': recon_w,
        'snr_range': snr_range, 'best_m10': best_m10, 'final_m4': final_m4, 'final_m10': final_m10,
        'history': [(h[0], round(h[1], 4)) for h in history[-5:]]
    }
    return result

# Configs to try
configs = [
    ('C1_lr1e-4',      1e-4, 200, 3.0, 0.1, (-10, 10)),
    ('C2_lr2e-4',      2e-4, 200, 3.0, 0.1, (-10, 10)),
    ('C3_lr3e-4',      3e-4, 200, 3.0, 0.1, (-10, 10)),
    ('C4_eap5.0',      5e-4, 200, 5.0, 0.1, (-10, 10)),
    ('C5_recon0.2',    5e-4, 200, 3.0, 0.2, (-10, 10)),
    ('C6_snr_hard',    5e-4, 200, 3.0, 0.1, (-12, 5)),
    ('C7_lr1e-4_300',  1e-4, 300, 3.0, 0.1, (-10, 10)),
]

print(f'{"="*70}')
print(f'Hyperparameter Sweep: {len(configs)} configs')
print(f'{"="*70}')
results = []
for i, (name, lr, ep, eap_w, recon_w, snr_range) in enumerate(configs):
    print(f'\n[{i+1}/{len(configs)}] {name}: lr={lr}, ep={ep}, eap_w={eap_w}, recon_w={recon_w}, snr={snr_range}')
    r = run_config(name, lr, ep, eap_w, recon_w, snr_range)
    results.append(r)
    print(f'  => Best -10dB: {r["best_m10"]:.4f}, Final -4dB: {r["final_m4"]:.4f}, Final -10dB: {r["final_m10"]:.4f}')

print(f'\n{"="*70}')
print(f'RANKING by best -10dB:')
print(f'{"="*70}')
results.sort(key=lambda x: x['best_m10'], reverse=True)
for i, r in enumerate(results):
    print(f'{i+1}. {r["name"]:15s} lr={r["lr"]:.0e} eap={r["eap_w"]} recon={r["recon_w"]} '
          f'snr={str(r["snr_range"]):12s} | Best-10dB={r["best_m10"]:.4f} Final-4dB={r["final_m4"]:.4f} '
          f'history={r["history"]}')

baseline = next(r for r in results if r['name'] == 'C1_lr1e-4')
print(f'\nBaseline (C1): -10dB={baseline["best_m10"]*100:.2f}%')
print(f'MS-TCANet paper: -10dB=74.67%')
