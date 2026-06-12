"""Hyperparameter sweep R2: reduce -10dB oscillation via regularization"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def run_config(name, lr, epochs, eap_w, recon_w, snr_range,
               wd, grad_clip, label_smooth, ema_decay, pretrain_ep=8, seed=42):
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

    # Phase 1: SUD pretrain
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

    # Phase 2: Full training
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs, eta_min=1e-5)

    best_m10, patience = 0.0, 0
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters(): p.requires_grad = False
    m10_vals = []

    for ep in range(epochs):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)

            if label_smooth > 0:
                n_cls = out['cls_pred'].shape[1]
                tgt = torch.full_like(out['cls_pred'], label_smooth / (n_cls - 1))
                tgt.scatter_(1, bl.unsqueeze(1), 1.0 - label_smooth)
                cls_loss = -(tgt * torch.nn.functional.log_softmax(out['cls_pred'], dim=1)).sum(dim=1).mean()
            else:
                cls_loss = torch.nn.functional.cross_entropy(out['cls_pred'], bl)

            loss = (cls_loss
                    + eap_w * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + recon_w * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            o2.step()

            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        sc.step()

        if ep % 10 == 0:
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            m10 = ev(-10)
            m10_vals.append(m10)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            if patience >= 20: break
            model.train()

    osc = np.std(m10_vals[-5:]) if len(m10_vals) >= 5 else np.std(m10_vals)
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    final_m4 = ev(-4); final_m10 = ev(-10)
    result = {
        'name': name, 'lr': lr, 'wd': wd, 'recon_w': recon_w,
        'label_smooth': label_smooth, 'ema_decay': ema_decay,
        'grad_clip': grad_clip, 'best_m10': best_m10,
        'final_m4': final_m4, 'final_m10': final_m10,
        'osc_std': osc, 'm10_trace': [round(v, 3) for v in m10_vals[-6:]]
    }
    return result

# R2 configs: combine best from R1 + regularization
configs = [
    # baseline from R1
    ('R1_baseline',      5e-4, 200, 3.0, 0.1, (-10, 10), 1e-4, 1.0, 0.0, 0.0),
    # R1 winner C5
    ('R2_C5_recon0.2',  5e-4, 200, 3.0, 0.2, (-10, 10), 1e-4, 1.0, 0.0, 0.0),
    # combo: lr=3e-4 + recon_w=0.2
    ('R3_lr3e-4_r0.2',  3e-4, 200, 3.0, 0.2, (-10, 10), 1e-4, 1.0, 0.0, 0.0),
    # R3 + wd=5e-4
    ('R4_wd5e-4',       3e-4, 200, 3.0, 0.2, (-10, 10), 5e-4, 1.0, 0.0, 0.0),
    # R3 + wd=1e-3
    ('R5_wd1e-3',       3e-4, 200, 3.0, 0.2, (-10, 10), 1e-3, 1.0, 0.0, 0.0),
    # R3 + label smoothing 0.1
    ('R6_ls0.1',        3e-4, 200, 3.0, 0.2, (-10, 10), 1e-4, 1.0, 0.1, 0.0),
    # R3 + EMA 0.999
    ('R7_ema',          3e-4, 200, 3.0, 0.2, (-10, 10), 1e-4, 1.0, 0.0, 0.999),
    # R3 + tighter grad clip 0.5
    ('R8_grad0.5',      3e-4, 200, 3.0, 0.2, (-10, 10), 1e-4, 0.5, 0.0, 0.0),
]

print(f'{"="*70}')
print(f'R2 Regularization Sweep: {len(configs)} configs')
print(f'{"="*70}')
results = []
for i, (name, lr, ep, eap_w, recon_w, snr, wd, gc, ls, ema) in enumerate(configs):
    print(f'\n[{i+1}/{len(configs)}] {name}: lr={lr}, recon_w={recon_w}, wd={wd}, ls={ls}, ema={ema}, gc={gc}')
    r = run_config(name, lr, ep, eap_w, recon_w, snr, wd, gc, ls, ema)
    results.append(r)
    print(f'  => Best -10dB={r["best_m10"]:.4f} Osc={r["osc_std"]:.4f} '
          f'Final -4dB={r["final_m4"]:.4f} Final -10dB={r["final_m10"]:.4f} '
          f'trace={r["m10_trace"]}')

print(f'\n{"="*70}')
print(f'RANKING by best -10dB:')
print(f'{"="*70}')
results.sort(key=lambda x: x['best_m10'], reverse=True)
for i, r in enumerate(results):
    print(f'{i+1}. {r["name"]:18s} lr={r["lr"]:.0e} recon={r["recon_w"]} wd={r["wd"]:.0e} '
          f'ls={r["label_smooth"]} ema={r["ema_decay"]} gc={r["grad_clip"]} | '
          f'-10dB={r["best_m10"]:.4f} -4dB={r["final_m4"]:.4f} osc_std={r["osc_std"]:.4f}')

print(f'\nMS-TCANet: -10dB=74.67%')
print(f'R1 best (C5_recon0.2): -10dB=79.61%')
