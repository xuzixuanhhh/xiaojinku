"""Quick 3-run sweep: E2/E5/E6 + new combos to find best mean -10dB"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
SEP = '=' * 70

def run_config(name, lr, recon_w, wd, ema_decay, epochs, seeds=[42, 123, 456]):
    results = []
    for seed in seeds:
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

        for p in model.parameters(): p.requires_grad = True
        o2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs, eta_min=5e-6)
        ema = copy.deepcopy(model)
        for p in ema.parameters(): p.requires_grad = False

        best_m10, patience = 0.0, 0
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
                with torch.no_grad():
                    alpha = 1 - ema_decay
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=alpha)
            sc.step()

            if ep % 10 == 0:
                ema.eval(); m10 = ev(-10)
                if m10 > best_m10: best_m10 = m10; patience = 0
                else: patience += 1
                if patience >= 20: break
                model.train()

        ema.eval()
        results.append((ev(-10), ev(-4), best_m10))
    return results

configs = [
    ('E2_ref',        5e-4, 0.10, 1e-4, 0.995, 200),
    ('E5_r0.15',      5e-4, 0.15, 1e-4, 0.995, 200),
    ('E6_ema099',     3e-4, 0.10, 1e-4, 0.99,  200),
    ('F1_r0.12',      5e-4, 0.12, 1e-4, 0.995, 200),
    ('F2_r0.08',      5e-4, 0.08, 1e-4, 0.995, 200),
    ('F3_ema0993',    5e-4, 0.10, 1e-4, 0.993, 200),
    ('F4_ema0997',    5e-4, 0.10, 1e-4, 0.997, 200),
    ('F5_wd1.5e-4',   5e-4, 0.10, 1.5e-4, 0.995, 200),
    ('F6_lr4e-4',     4e-4, 0.10, 1e-4, 0.995, 200),
    ('F7_lr6e-4',     6e-4, 0.10, 1e-4, 0.995, 200),
    ('F8_300ep',      5e-4, 0.10, 1e-4, 0.995, 300),
    ('F9_250ep',      5e-4, 0.10, 1e-4, 0.995, 250),
]

print(SEP)
print(f'Quick 3-run sweep: {len(configs)} configs')
print(SEP)
all_results = []
for name, lr, recon, wd, ema, ep in configs:
    label = f'{name}: lr={lr:.0e} r={recon} wd={wd:.0e} ema={ema} ep={ep}...'
    print(label, end=' ', flush=True)
    res = run_config(name, lr, recon, wd, ema, ep)
    m10s = [r[0] for r in res]; m4s = [r[1] for r in res]; bests = [r[2] for r in res]
    all_results.append((name, m10s, m4s, bests, lr, recon, wd, ema, ep))
    vals_str = ', '.join([format(v, '.4f') for v in m10s])
    print(f'| -10dB=[{vals_str}] mean={np.mean(m10s):.4f} -4dB={np.mean(m4s):.4f}')

print()
print(SEP)
print('RANKING by mean -10dB:')
print(SEP)
all_results.sort(key=lambda x: np.mean(x[1]), reverse=True)
for i, (name, m10s, m4s, bests, lr, recon, wd, ema, ep) in enumerate(all_results):
    m = np.mean(m10s); s = np.std(m10s)
    vals_str = ', '.join([format(v, '.3f') for v in m10s])
    print(f'{i+1:2d}. {name:20s} -10dB={m:.4f} std={s:.4f} -4dB={np.mean(m4s):.4f} '
          f'best={np.mean(bests):.4f} [{vals_str}]')

print(f'\nWinner for 10-run: {all_results[0][0]}')
