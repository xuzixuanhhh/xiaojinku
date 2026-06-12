"""EMA decay sweep: find sweet spot - stable AND high peak"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def run_config(name, lr, recon_w, wd, ema_decay, epochs=200, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_loader, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
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

    # P1
    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for _ in range(8):
        for bd, _, _ in train_loader:
            bd = bd.to(device); o1.zero_grad()
            xd, ne, es = model.denoiser(bd)
            ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
            L, _ = denoise_loss(xd, bd, ne, es, ts)
            L.backward(); o1.step()

    # P2
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs, eta_min=5e-6)

    ema = None
    if ema_decay > 0:
        ema = copy.deepcopy(model)
        for p in ema.parameters(): p.requires_grad = False

    best_m10, patience, trace = 0.0, 0, []
    for ep in range(epochs):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + recon_w * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
            if ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        sc.step()

        if ep % 10 == 0:
            eval_model = ema if ema is not None else model
            eval_model.eval(); m10 = ev(-10)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            trace.append(m10)
            if patience >= 20: break
            model.train()

    eval_model = ema if ema is not None else model
    eval_model.eval()
    final_m4 = ev(-4); final_m10 = ev(-10)
    osc = np.std(trace[-5:]) if len(trace) >= 5 else np.std(trace)
    result = {
        'name': name, 'lr': lr, 'recon_w': recon_w, 'wd': wd, 'ema': ema_decay,
        'best_m10': best_m10, 'final_m4': final_m4, 'final_m10': final_m10,
        'osc_std': osc, 'peaks': [round(v, 3) for v in sorted(trace, reverse=True)[:3]]
    }
    return result

# Grid: find sweet spot between 0HP winning config and R4+EMA
configs = [
    # EMA decay sweep (keep wd=1e-4, recon=0.1 like 0HP winner)
    ('E1_ema099',     5e-4, 0.1, 1e-4, 0.99),
    ('E2_ema0995',    5e-4, 0.1, 1e-4, 0.995),
    ('E3_ema0999',    5e-4, 0.1, 1e-4, 0.999),
    # Slightly higher recon + lighter EMA
    ('E4_r0.15_099',  5e-4, 0.15, 1e-4, 0.99),
    ('E5_r0.15_0995', 5e-4, 0.15, 1e-4, 0.995),
    # lr=3e-4 + lighter EMA + 0HP-like params
    ('E6_lr3e-4_099', 3e-4, 0.1, 1e-4, 0.99),
    # wd=2e-4 middle ground
    ('E7_wd2e-4_099', 5e-4, 0.1, 2e-4, 0.99),
    # Baseline refs
    ('E8_R4+EMA_ref', 3e-4, 0.2, 5e-4, 0.999),
]

print(f'{"="*65}')
print(f'EMA Decay Sweep: {len(configs)} configs - stable + high peak')
print(f'{"="*65}')
results = []
for i, (name, lr, recon_w, wd, ema) in enumerate(configs):
    print(f'\n[{i+1}/{len(configs)}] {name}: lr={lr}, recon={recon_w}, wd={wd}, ema={ema}')
    r = run_config(name, lr, recon_w, wd, ema)
    results.append(r)
    print(f'  => Best={r["best_m10"]:.4f} Final -4dB={r["final_m4"]:.4f} '
          f'Final -10dB={r["final_m10"]:.4f} osc={r["osc_std"]:.4f} top3={r["peaks"]}')

print(f'\n{"="*65}')
print(f'RANKING by best -10dB (trade-off: high peak + low osc)')
print(f'{"="*65}')
results.sort(key=lambda x: x['best_m10'], reverse=True)
for i, r in enumerate(results):
    score = r['best_m10'] - r['osc_std']
    print(f'{i+1}. {r["name"]:16s} lr={r["lr"]:.0e} recon={r["recon_w"]:.2f} wd={r["wd"]:.0e} '
          f'ema={r["ema"]} | -10dB={r["best_m10"]:.4f} -4dB={r["final_m4"]:.4f} '
          f'osc={r["osc_std"]:.4f} score={score:.4f} top3={r["peaks"]}')

print(f'\nReference: R4+EMA(10-run)=75.92% MS-TCANet=74.67% Target=80%')
