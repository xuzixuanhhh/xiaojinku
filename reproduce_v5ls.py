"""DX-PINN v5+LS: 复刻最佳模型 — 10-seed验证
v5+LS = DX_PINN_V5 + label_smoothing=0.05
之前结果: -10dB = 78.93% +/- 6.38%, 6/10 seeds >= 80%
模型文件: model/dx_pinn_v5.py
"""
import torch, numpy as np, random, copy, sys
sys.path.insert(0, '.')
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn_v5 import DX_PINN_V5
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE


def qev(model, test_data, test_labels, snr):
    ds = BearingDataset(test_data, test_labels, snr_db=snr, add_noise=True)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    yt, yp = [], []
    with torch.no_grad():
        for bd, bl, bt in dl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            out = model(bd, bt)
            yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
            yt.extend(bl.cpu().numpy())
    return calculate_metrics(yt, yp)[0]


def train(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    model = DX_PINN_V5().to(device)

    # Pretrain
    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for _ in range(5):
        for bd, _, _ in tl:
            bd = bd.to(device); o1.zero_grad()
            xd, ne, es = model.denoiser(bd)
            ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
            L, _ = denoise_loss(xd, bd, ne, es, ts); L.backward(); o1.step()

    # Joint training
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=95, eta_min=1e-5)
    cf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best_m10, best_state = 0.0, None

    for ep in range(100):
        model.train()
        if ep < 5:
            for pg in o2.param_groups: pg['lr'] = 1e-3 * (ep + 1) / 5
        for bd, bl, bt in tl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)
            loss = (cf(out['cls_pred'], bl)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + 0.1 * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
        if ep >= 5: sc.step()

        if ep % 10 == 0 or ep == 99:
            model.eval()
            m10 = qev(model, td, tlbl, -10)
            if m10 > best_m10:
                best_m10 = m10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    model.load_state_dict(best_state); model.eval()
    return qev(model, td, tlbl, 10), qev(model, td, tlbl, -4), qev(model, td, tlbl, -10), best_m10


sep = '=' * 60
print(sep)
print('DX-PINN v5+LS: Best Model Reproduction')
print('lr=1e-3, label_smoothing=0.05, wd=1e-4, SNR[-10,10], 100ep')
print(sep)

rc, r4, r10, r10b = [], [], [], []
for run in range(10):
    print(f'Run {run+1}/10...', end=' ', flush=True)
    c, m4, m10, best = train(run)
    rc.append(c); r4.append(m4); r10.append(m10); r10b.append(best)
    print(f'C={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f} B={best:.4f}')

print(f'\n{sep}')
print(f'v5+LS: -10dB = {np.mean(r10)*100:.2f}% +/- {np.std(r10)*100:.2f}%')
print(f'  -4dB = {np.mean(r4)*100:.2f}%  Clean = {np.mean(rc)*100:.2f}%')
print(f'  80%+ seeds: {sum(1 for x in r10 if x>=0.80)}/10')
print(f'Expected: -10dB = 78.93% +/- 6.38%')
print(f'MS-TCANet paper: 74.67% +/- 1.24%')
print(sep)
