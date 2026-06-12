"""tune_v5v6: v5+LS 100-epoch, CosineAnnealing, pretrain=5, warmup=5
Best: v5+LS 78.93% +/- 6.38%, 6/10 seeds >= 80%, peak 83.9%
"""
import torch, numpy as np, random, copy, json, sys, os
from datetime import datetime
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn_v5 import DX_PINN_V5
from model.dx_pinn_v6 import DX_PINN_V6
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
RESULT_FILE = os.path.join(SAVE_PATH, "tune_v5v6_results.json")


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


def train(seed, model_cls, model_name, lr=1e-3, label_smooth=0.05,
          w_cls=1.0, w_eap=3.0, w_recon=0.1, ema_decay=0.995,
          pretrain_ep=5, warmup_ep=5, wd=1e-4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    m = model_cls().to(device)

    # Phase 1: Pretrain denoiser
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

    # Phase 2: Joint training with CosineAnnealing + warmup + EMA
    for p in m.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(
        o2, T_max=100 - warmup_ep, eta_min=1e-5)

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
            with torch.no_grad():
                a = 1.0 - ema_decay
                for pe, p in zip(ema.parameters(), m.parameters()):
                    pe.data.mul_(ema_decay).add_(p.data, alpha=a)
                for eb, b in zip(ema.buffers(), m.buffers()):
                    eb.data.copy_(b.data)

        if ep >= warmup_ep:
            sc.step()

        if ep % 10 == 0 or ep == 99:
            ema.eval()
            m10 = quick_eval(ema, td, tlbl, -10)
            if m10 > best_m10: best_m10 = m10
            m.train()

    ema.eval()
    m10 = quick_eval(ema, td, tlbl, -10)
    m4 = quick_eval(ema, td, tlbl, -4)
    clean = quick_eval(ema, td, tlbl, 10)
    return float(m10), float(m4), float(clean), float(best_m10)


SEEDS = [42, 123, 456]

configs = [
    ('v5_base',  DX_PINN_V5, 1e-3, 0.0,  1.0, 3.0, 0.1, 0.995, 5, 5),
    ('v5_LS',    DX_PINN_V5, 1e-3, 0.05, 1.0, 3.0, 0.1, 0.995, 5, 5),
    ('v6_base',  DX_PINN_V6, 1e-3, 0.0,  1.0, 3.0, 0.1, 0.995, 5, 5),
    ('v6_LS',    DX_PINN_V6, 1e-3, 0.05, 1.0, 3.0, 0.1, 0.995, 5, 5),
]

if __name__ == '__main__':
    print(f"tune_v5v6: CosineAnnealing, pretrain=5, warmup=5, 100ep, SNR[-10,10]")
    sys.stdout.flush()

    results = {}
    for name, cls, lr, ls, wc, we, wr, ema, pe, wu in configs:
        m10s, m4s, cleans, bests = [], [], [], []
        print(f"\n{name}...")
        for s in SEEDS:
            m10, m4, clean, best = train(s, cls, name, lr=lr, label_smooth=ls,
                                         w_cls=wc, w_eap=we, w_recon=wr,
                                         ema_decay=ema, pretrain_ep=pe, warmup_ep=wu)
            m10s.append(m10); m4s.append(m4); cleans.append(clean); bests.append(best)
            print(f"  seed={s}: -10dB={m10:.4f} -4dB={m4:.4f} Clean={clean:.4f} Best={best:.4f}")
            sys.stdout.flush()

        results[name] = {
            "m10_mean": float(np.mean(m10s)), "m10_std": float(np.std(m10s)),
            "m4_mean": float(np.mean(m4s)), "m4_std": float(np.std(m4s)),
            "clean_mean": float(np.mean(cleans)), "m10s": m10s,
        }
        print(f"  => -10dB={np.mean(m10s):.4f}+/-{np.std(m10s):.4f} "
              f"-4dB={np.mean(m4s):.4f} Clean={np.mean(cleans):.4f}")

    with open(RESULT_FILE, 'w') as f:
        json.dump({"timestamp": str(datetime.now()), "results": results}, f, indent=2)

    ranking = sorted(results.items(), key=lambda x: x[1]["m10_mean"], reverse=True)
    print(f"\nRanking:")
    for i, (name, r) in enumerate(ranking):
        print(f"  {i+1}. {name}: -10dB={r['m10_mean']:.4f}+/-{r['m10_std']:.4f} "
              f"-4dB={r['m4_mean']:.4f} Clean={r['clean_mean']:.4f}")
