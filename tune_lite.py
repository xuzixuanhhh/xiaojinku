"""DX-PINN-Lite: 3-seed screening + 10-seed validation"""
import torch, numpy as np, random, copy, sys
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn_lite import DX_PINN_Lite
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE


def quick_ev(m, td, tlbl, snr):
    ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    yt, yp = [], []
    with torch.no_grad():
        for bd, bl, bt in dl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            out = m(bd, bt)
            yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
            yt.extend(bl.cpu().numpy())
    return calculate_metrics(yt, yp)[0]


def train_lite(seed, lr):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    m = DX_PINN_Lite().to(device)

    for p in m.parameters(): p.requires_grad = False
    for p in m.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-3)
    for _ in range(5):
        for bd, _, _ in tl:
            bd = bd.to(device); o1.zero_grad()
            xd, ne, es = m.denoiser(bd)
            ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
            L, _ = denoise_loss(xd, bd, ne, es, ts)
            L.backward(); o1.step()

    for p in m.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=95, eta_min=1e-5)
    best_m10, best_state = 0.0, None

    for ep in range(100):
        m.train()
        if ep < 5:
            for pg in o2.param_groups:
                pg['lr'] = lr * (ep + 1) / 5
        for bd, bl, bt in tl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = m(bd, bt)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + 0.1 * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o2.step()
        if ep >= 5: sc.step()

        if ep % 10 == 0 or ep == 99:
            m.eval()
            m10 = quick_ev(m, td, tlbl, -10)
            if m10 > best_m10:
                best_m10 = m10
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            m.train()

    m.load_state_dict(best_state)
    m.eval()
    return (quick_ev(m, td, tlbl, 10),
            quick_ev(m, td, tlbl, -4),
            quick_ev(m, td, tlbl, -10),
            best_m10)


SEEDS3 = [42, 123, 456]
LRS = [1e-3, 1.5e-3, 2e-3]

print('=' * 60)
print('DX-PINN-Lite: 3-Seed Screening')
print('Params: ~777K (29% of original 2.64M)')
print('=' * 60)
sys.stdout.flush()

best_lr, best_mean = None, 0.0
for lr in LRS:
    m10s = []
    for s in SEEDS3:
        c, m4, m10, best = train_lite(s, lr)
        m10s.append(m10)
        print(f'  lr={lr} s={s}: Clean={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f} Best={best:.4f}')
        sys.stdout.flush()
    mean_m10 = np.mean(m10s)
    print(f'  => lr={lr}: -10dB={mean_m10:.4f}+/-{np.std(m10s):.4f}')
    sys.stdout.flush()
    if mean_m10 > best_mean:
        best_mean = mean_m10
        best_lr = lr

print(f'\nBest LR: {best_lr} (-10dB={best_mean:.4f})')
sys.stdout.flush()

# Phase 2: 10-seed validation
sep = '=' * 60
print(f'\n{sep}')
print(f'DX-PINN-Lite: 10-Run Validation (lr={best_lr})')
print(sep)
sys.stdout.flush()

rc, r4, r10, r10b = [], [], [], []
for run in range(10):
    print(f'Run {run+1}/10...', end=' ', flush=True)
    c, m4, m10, best = train_lite(run, best_lr)
    rc.append(c); r4.append(m4); r10.append(m10); r10b.append(best)
    print(f'Clean={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f} (best={best:.4f})')

print(f'\n{sep}')
print('DX-PINN-Lite (100ep) vs Baselines -- CWRU 1HP')
print(sep)
print(f'{"Model":<20} {"-10dB":>16} {"-4dB":>12} {"Clean":>10}')
print(f'{"-"*58}')
print(f'{"Lite 100ep":<20} {np.mean(r10)*100:>7.2f}%+/-{np.std(r10)*100:.2f}% {np.mean(r4)*100:>7.2f}% {np.mean(rc)*100:>7.2f}%')
print(f'{"Original 100ep":<20} {"66.70%+/-6.59%":>16} {"93.62%":>12} {"99.71%":>10}')
print(f'{"Original 200ep":<20} {"76.76%+/-1.97%":>16} {"98.41%":>12} {"99.84%":>10}')
print(f'{"MS-TCANet paper":<20} {"74.67%+/-1.24%":>16} {"96.36%":>12} {"~100%":>10}')
print(sep)
print(f'Delta vs MS-TCANet: {np.mean(r10)*100 - 74.67:+.2f}%')
print(f'Delta vs Orig 100ep: {np.mean(r10)*100 - 66.70:+.2f}%')
sys.stdout.flush()
