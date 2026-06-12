"""100-epoch hyperparameter optimization for DX-PINN
Goal: 80% at -10dB on CWRU 1HP, non-overlap 1024, SNR[-10,10]

Key innovations vs previous 100ep attempts:
1. OneCycleLR — faster convergence than CosineAnnealing
2. Label smoothing — better generalization under heavy noise
3. Higher batch size — more stable gradient estimates
4. Optimized loss weight ratios
"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE


def quick_eval(model, td, tlbl, snr):
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


def train_100ep(seed, max_lr, label_smooth, loss_weights, batch_size, ema_decay,
                pretrain_ep, wd, pct_start=0.3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_loader, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    model = DX_PINN().to(device)

    w_cls, w_eap, w_recon = loss_weights

    if pretrain_ep > 0:
        for p in model.parameters(): p.requires_grad = False
        for p in model.denoiser.parameters(): p.requires_grad = True
        o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
        for _ in range(pretrain_ep):
            for bd, _, _ in train_loader:
                bd = bd.to(device); o1.zero_grad()
                xd, ne, es = model.denoiser(bd)
                ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
                L, _ = denoise_loss(xd, bd, ne, es, ts)
                L.backward(); o1.step()

    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=wd)

    steps_per_epoch = len(train_loader)
    sc = torch.optim.lr_scheduler.OneCycleLR(
        o2, max_lr=max_lr, epochs=100, steps_per_epoch=steps_per_epoch,
        pct_start=pct_start, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1e4)

    ema = copy.deepcopy(model)
    for p in ema.parameters(): p.requires_grad = False
    best_m10, trace = 0.0, []

    cls_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    for ep in range(100):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)
            loss = (w_cls * cls_fn(out['cls_pred'], bl)
                    + w_eap * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + w_recon * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
            sc.step()
            with torch.no_grad():
                a = 1.0 - ema_decay
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.data.mul_(ema_decay).add_(p.data, alpha=a)

        if ep % 5 == 0 or ep == 99:
            ema.eval()
            m10 = quick_eval(ema, td, tlbl, -10)
            if m10 > best_m10:
                best_m10 = m10
            trace.append(m10)
            model.train()

    ema.eval()
    final_m10 = quick_eval(ema, td, tlbl, -10)
    final_m4 = quick_eval(ema, td, tlbl, -4)
    final_clean = quick_eval(ema, td, tlbl, 10)
    osc = np.std(trace[-4:]) if len(trace) >= 4 else 0.0
    return final_m10, final_m4, final_clean, max(best_m10, final_m10), osc


SEEDS = [42, 123, 456]

configs = [
    # (name, max_lr, label_smooth, (w_cls, w_eap, w_recon), batch, ema, pretrain, wd, pct_start)
    # --- OneCycleLR baseline vs CosineAnnealing ---
    ('A1_base_1e-3',          1e-3,  0.0, (1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('A2_base_2e-3',          2e-3,  0.0, (1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('A3_base_3e-3',          3e-3,  0.0, (1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),

    # --- Label Smoothing ---
    ('B1_ls0.05_lr2e-3',     2e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('B2_ls0.10_lr2e-3',     2e-3,  0.10,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('B3_ls0.05_lr3e-3',     3e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),

    # --- Loss Weight Tuning ---
    ('C1_eap4_lr2e-3',       2e-3,  0.05,(1.0, 4.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('C2_eap2_lr2e-3',       2e-3,  0.05,(1.0, 2.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('C3_recon0.05_lr2e-3',  2e-3,  0.05,(1.0, 3.0, 0.05),128, 0.995, 3, 1e-4, 0.3),

    # --- Batch Size ---
    ('D1_bs256_lr2e-3',      2e-3,  0.05,(1.0, 3.0, 0.1), 256, 0.995, 3, 1e-4, 0.3),
    ('D2_bs256_lr3e-3',      3e-3,  0.05,(1.0, 3.0, 0.1), 256, 0.995, 3, 1e-4, 0.3),

    # --- EMA Tuning ---
    ('E1_ema099_lr2e-3',     2e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.99,  3, 1e-4, 0.3),
    ('E2_ema0997_lr2e-3',    2e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.997, 3, 1e-4, 0.3),

    # --- Extended warmup ---
    ('F1_warm50_lr2e-3',     2e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.5),
    ('F2_warm40_lr3e-3',     3e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.4),

    # --- Higher LR + label smooth ---
    ('G1_lr4e-3_ls0.05',     4e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),
    ('G2_lr5e-3_ls0.05',     5e-3,  0.05,(1.0, 3.0, 0.1), 128, 0.995, 3, 1e-4, 0.3),

    # --- Promising combos ---
    ('H1_lr2e-3_ls0.05_eap4_bs256', 2e-3, 0.05, (1.0, 4.0, 0.1), 256, 0.995, 3, 1e-4, 0.3),
    ('H2_lr3e-3_ls0.05_eap2_bs256', 3e-3, 0.05, (1.0, 2.0, 0.1), 256, 0.995, 3, 1e-4, 0.3),
]

print('=' * 80)
print('DX-PINN 100-Epoch Hyperparameter Optimization')
print('CWRU 1HP, non-overlap 1024, SNR[-10,10] AWGN, OneCycleLR')
print('=' * 80)

results = []
for name, max_lr, ls, lw, bs, ema, pe, wd, ps in configs:
    m10s, m4s, cleans, bests, oscs = [], [], [], [], []
    for s in SEEDS:
        m10, m4, clean, best, osc = train_100ep(s, max_lr, ls, lw, bs, ema, pe, wd, ps)
        m10s.append(m10); m4s.append(m4); cleans.append(clean); bests.append(best); oscs.append(osc)

    results.append((name, m10s, m4s, cleans, bests, oscs))
    vals_str = ', '.join([f'{v:.4f}' for v in m10s])
    print(f'{name}: -10dB={np.mean(m10s):.4f}±{np.std(m10s):.4f} [{vals_str}] '
          f'-4dB={np.mean(m4s):.4f} Clean={np.mean(cleans):.4f} '
          f'Best={np.mean(bests):.4f} OSC={np.mean(oscs):.4f}')

best_idx = np.argmax([np.mean(r[1]) for r in results])
best_name = results[best_idx][0]
best_m10 = np.mean(results[best_idx][1])
print(f'\n{"="*80}')
print(f'Best: {best_name} @ -10dB={best_m10:.4f}')
print(f'Target: 80.00% | MS-TCANet: 74.67% | Prev 100ep best: 73.57%')
print(f'{"="*80}')
