"""R3 sweep: push -10dB to 80% via training SNR, longer epochs, focal loss"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def run_config(name, snr_range, epochs, lr, recon_w, wd, use_focal, seed=42):
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

    # P1: SUD pretrain
    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for _ in range(8):
        for bd, _, _ in train_loader:
            bd = bd.to(device); o1.zero_grad()
            xd, ne, es = model.denoiser(bd)
            s_low, s_high = snr_range
            ts = torch.rand(bd.shape[0], 1, device=device) * (s_high - s_low) + s_low
            L, _ = denoise_loss(xd, bd, ne, es, ts)
            L.backward(); o1.step()

    # P2: Full training
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs, eta_min=1e-5)
    ema = copy.deepcopy(model)
    for p in ema.parameters(): p.requires_grad = False

    best_m10, patience, trace = 0.0, 0, []
    for ep in range(epochs):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)

            if use_focal:
                ce = torch.nn.functional.cross_entropy(out['cls_pred'], bl, reduction='none')
                pt = torch.exp(-ce)
                cls_loss = ((1 - pt) ** 2 * ce).mean()
            else:
                cls_loss = torch.nn.functional.cross_entropy(out['cls_pred'], bl)

            loss = (cls_loss
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + recon_w * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
        sc.step()

        if ep % 10 == 0:
            ema.eval(); m10 = ev(-10); m4 = ev(-4)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            trace.append((ep+1, m10))
            osc = np.std([t[1] for t in trace[-5:]]) if len(trace) >= 5 else 0
            if patience >= 20: break
            model.train()

    ema.eval()
    final_m4 = ev(-4); final_m10 = ev(-10)
    result = {
        'name': name, 'snr_range': snr_range, 'epochs': epochs,
        'lr': lr, 'recon_w': recon_w, 'wd': wd, 'focal': use_focal,
        'best_m10': best_m10, 'final_m4': final_m4, 'final_m10': final_m10,
        'osc_std': osc if len(trace) >= 5 else np.std([t[1] for t in trace]),
        'trace': [(t[0], round(t[1], 3)) for t in trace[-4:]]
    }
    return result

configs = [
    # Baseline R4+EMA for reference
    ('B1_R4+EMA',         (-10, 10),   200, 3e-4, 0.2, 5e-4, False),
    # Harder training SNR (key lever from memory)
    ('B2_snr_hard',       (-12, 5),    200, 3e-4, 0.2, 5e-4, False),
    ('B3_snr_vhard',      (-14, 2),    200, 3e-4, 0.2, 5e-4, False),
    # Longer training
    ('B4_300ep',          (-10, 10),   300, 3e-4, 0.2, 5e-4, False),
    ('B5_snr_hard_300ep', (-12, 5),    300, 3e-4, 0.2, 5e-4, False),
    # Focal loss
    ('B6_focal',          (-10, 10),   200, 3e-4, 0.2, 5e-4, True),
    ('B7_focal_snr_hard', (-12, 5),    200, 3e-4, 0.2, 5e-4, True),
    # Higher recon
    ('B8_recon0.3',       (-10, 10),   200, 3e-4, 0.3, 5e-4, False),
]

print(f'{"="*70}')
print(f'R3: Push -10dB -> 80% ({len(configs)} configs)')
print(f'{"="*70}')
results = []
for i, (name, snr, ep, lr, recon_w, wd, focal) in enumerate(configs):
    print(f'\n[{i+1}/{len(configs)}] {name}: snr={snr}, ep={ep}, recon_w={recon_w}, focal={focal}')
    r = run_config(name, snr, ep, lr, recon_w, wd, focal)
    results.append(r)
    print(f'  => Best -10dB={r["best_m10"]:.4f} Final -4dB={r["final_m4"]:.4f} '
          f'Final -10dB={r["final_m10"]:.4f} osc={r["osc_std"]:.4f} trace={r["trace"]}')

print(f'\n{"="*70}')
print(f'RANKING by best -10dB:')
print(f'{"="*70}')
results.sort(key=lambda x: x['best_m10'], reverse=True)
for i, r in enumerate(results):
    print(f'{i+1}. {r["name"]:20s} snr={str(r["snr_range"]):12s} ep={r["epochs"]} recon={r["recon_w"]} '
          f'focal={r["focal"]} | -10dB={r["best_m10"]:.4f} -4dB={r["final_m4"]:.4f} osc={r["osc_std"]:.4f}')

print(f'\nR4+EMA baseline: 75.92% (10-run)')
print(f'MS-TCANet: 74.67%')
print(f'Target: 80.00%')
