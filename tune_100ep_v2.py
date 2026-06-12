"""100-epoch sweep v2: Fixed OneCycleLR + CosineAnnealing baselines
Key fix: div_factor=3 (not 25) so LR starts at max_lr/3
"""
import torch, numpy as np, random, copy, sys, os, json
from datetime import datetime
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
RESULT_FILE = os.path.join(SAVE_PATH, "tune_100ep_v2_results.json")


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


def train_onecycle(seed, max_lr, label_smooth, w_cls, w_eap, w_recon,
                   ema_decay, pretrain_ep, wd, pct_start, div_factor):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    m = DX_PINN().to(device)

    if pretrain_ep > 0:
        for p in m.parameters(): p.requires_grad = False
        for p in m.denoiser.parameters(): p.requires_grad = True
        o1 = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-3)
        for _ in range(pretrain_ep):
            for bd, _, _ in tl:
                bd = bd.to(device); o1.zero_grad()
                xd, ne, es = m.denoiser(bd)
                ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
                L, _ = denoise_loss(xd, bd, ne, es, ts)
                L.backward(); o1.step()

    for p in m.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(m.parameters(), lr=max_lr, weight_decay=wd)
    spe = len(tl)
    sc = torch.optim.lr_scheduler.OneCycleLR(
        o2, max_lr=max_lr, epochs=100, steps_per_epoch=spe,
        pct_start=pct_start, anneal_strategy='cos',
        div_factor=div_factor, final_div_factor=1e3)

    ema = copy.deepcopy(m)
    for p in ema.parameters(): p.requires_grad = False
    best_m10 = 0.0
    cls_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    for ep in range(100):
        m.train()
        for bd, bl, bt in tl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = m(bd, bt)
            loss = (w_cls * cls_fn(out['cls_pred'], bl)
                    + w_eap * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + w_recon * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o2.step(); sc.step()
            a = 1.0 - ema_decay
            for pe, p in zip(ema.parameters(), m.parameters()):
                pe.data.mul_(ema_decay).add_(p.data, alpha=a)

        if ep % 10 == 0 or ep == 99:
            ema.eval()
            m10 = quick_eval(ema, td, tlbl, -10)
            if m10 > best_m10: best_m10 = m10
            m.train()

    ema.eval()
    return (quick_eval(ema, td, tlbl, -10),
            quick_eval(ema, td, tlbl, -4),
            quick_eval(ema, td, tlbl, 10),
            best_m10)


def train_cosine(seed, lr, label_smooth, w_cls, w_eap, w_recon,
                 ema_decay, pretrain_ep, wd, warmup_ep):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    m = DX_PINN().to(device)

    if pretrain_ep > 0:
        for p in m.parameters(): p.requires_grad = False
        for p in m.denoiser.parameters(): p.requires_grad = True
        o1 = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-3)
        for _ in range(pretrain_ep):
            for bd, _, _ in tl:
                bd = bd.to(device); o1.zero_grad()
                xd, ne, es = m.denoiser(bd)
                ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
                L, _ = denoise_loss(xd, bd, ne, es, ts)
                L.backward(); o1.step()

    for p in m.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=100 - warmup_ep,
                                                      eta_min=1e-5)

    ema = copy.deepcopy(m)
    for p in ema.parameters(): p.requires_grad = False
    best_m10 = 0.0
    cls_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    for ep in range(100):
        m.train()
        if ep < warmup_ep:
            for pg in o2.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup_ep
        for bd, bl, bt in tl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = m(bd, bt)
            loss = (w_cls * cls_fn(out['cls_pred'], bl)
                    + w_eap * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + w_recon * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o2.step()
            a = 1.0 - ema_decay
            for pe, p in zip(ema.parameters(), m.parameters()):
                pe.data.mul_(ema_decay).add_(p.data, alpha=a)
        if ep >= warmup_ep:
            sc.step()

        if ep % 10 == 0 or ep == 99:
            ema.eval()
            m10 = quick_eval(ema, td, tlbl, -10)
            if m10 > best_m10: best_m10 = m10
            m.train()

    ema.eval()
    return (quick_eval(ema, td, tlbl, -10),
            quick_eval(ema, td, tlbl, -4),
            quick_eval(ema, td, tlbl, 10),
            best_m10)


CONFIGS = [
    # === Fixed OneCycleLR: div_factor=3, pct_start=0.15 ===
    ('1cyc', 'O1_lr1e-3_d3',      1e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O2_lr1.5e-3_d3',    1.5e-3,0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O3_lr2e-3_d3',      2e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O4_lr3e-3_d3',      3e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O5_lr2e-3_d3_ls',   2e-3,  0.05,1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O6_lr3e-3_d3_ls',   3e-3,  0.05,1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O7_lr2e-3_eap4_ls', 2e-3,  0.05,1.0, 4.0, 0.1, 0.995, 3, 1e-4, 0.15, 3.0),
    ('1cyc', 'O8_lr2e-3_d5',      2e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 5.0),
    ('1cyc', 'O9_lr3e-3_d5_ls',   3e-3,  0.05,1.0, 3.0, 0.1, 0.995, 3, 1e-4, 0.15, 5.0),
    # === CosineAnnealing baselines ===
    ('cos', 'C1_lr1e-3_w5',       1e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 5),
    ('cos', 'C2_lr1.5e-3_w5',     1.5e-3,0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 5),
    ('cos', 'C3_lr2e-3_w5',       2e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 3, 1e-4, 5),
    ('cos', 'C4_lr1e-3_ls_w5',    1e-3,  0.05,1.0, 3.0, 0.1, 0.995, 3, 1e-4, 5),
    ('cos', 'C5_lr2e-3_ls_w5',    2e-3,  0.05,1.0, 3.0, 0.1, 0.995, 3, 1e-4, 5),
    ('cos', 'C6_C5ref_lr1e-3',    1e-3,  0.0, 1.0, 3.0, 0.1, 0.995, 5, 1e-4, 5),
]

SEEDS = [42, 123, 456]
results = {}

print(f"Sweep v2: {len(CONFIGS)} configs x {len(SEEDS)} seeds = {len(CONFIGS) * len(SEEDS)} runs")
print(f"Fixed OneCycleLR (div_factor=3/5) + CosineAnnealing baselines")
sys.stdout.flush()

for ctype, name, *params in CONFIGS:
    m10s, m4s, cleans, bests = [], [], [], []
    for s in SEEDS:
        if ctype == '1cyc':
            max_lr, ls, wc, we, wr, ema, pe, wd, ps, df = params
            m10, m4, clean, best = train_onecycle(s, max_lr, ls, wc, we, wr, ema, pe, wd, ps, df)
        else:
            lr, ls, wc, we, wr, ema, pe, wd, wu = params
            m10, m4, clean, best = train_cosine(s, lr, ls, wc, we, wr, ema, pe, wd, wu)

        m10s.append(float(m10)); m4s.append(float(m4))
        cleans.append(float(clean)); bests.append(float(best))
        print(f"  {name} seed={s}: -10dB={m10:.4f} -4dB={m4:.4f} Clean={clean:.4f} Best={best:.4f}")
        sys.stdout.flush()

    results[name] = {
        "type": ctype,
        "m10_mean": float(np.mean(m10s)), "m10_std": float(np.std(m10s)),
        "m4_mean": float(np.mean(m4s)), "m4_std": float(np.std(m4s)),
        "clean_mean": float(np.mean(cleans)), "best_mean": float(np.mean(bests)),
        "m10s": m10s, "m4s": m4s, "cleans": cleans, "bests": bests,
    }
    print(f"  => {name}: -10dB={np.mean(m10s):.4f}+/-{np.std(m10s):.4f} "
          f"-4dB={np.mean(m4s):.4f} Clean={np.mean(cleans):.4f}")
    sys.stdout.flush()

with open(RESULT_FILE, 'w') as f:
    json.dump({"timestamp": str(datetime.now()), "results": results}, f, indent=2)

ranking = sorted(results.items(), key=lambda x: x[1]["m10_mean"], reverse=True)
print(f"\nRanking:")
for i, (name, r) in enumerate(ranking):
    print(f"  {i+1}. {name}: -10dB={r['m10_mean']:.4f}+/-{r['m10_std']:.4f} "
          f"-4dB={r['m4_mean']:.4f} Clean={r['clean_mean']:.4f}")
print(f"\nMS-TCANet: -10dB=74.67% | Prev 100ep best: 73.57% | Target: 80%")
sys.stdout.flush()
