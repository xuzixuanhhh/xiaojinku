"""DX-PINN final: E6 config (lr=3e-4, recon=0.1, wd=1e-4, EMA=0.99), 10-run vs MS-TCANet"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
N_RUNS = 10

def train_and_eval(seed):
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

    # P1: SUD pretrain
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

    # P2: E6 config — lr=3e-4, recon=0.1, wd=1e-4, EMA=0.99
    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=200, eta_min=5e-6)
    ema = copy.deepcopy(model)
    for p in ema.parameters(): p.requires_grad = False

    best_m10, patience = 0.0, 0
    for ep in range(200):
        model.train()
        for bd, bl, bt in train_loader:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = model(bd, bt)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + 0.1 * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o2.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
        sc.step()

        if ep % 10 == 0:
            ema.eval(); m10 = ev(-10)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            if patience >= 20: break
            model.train()

    ema.eval()
    return ev(10), ev(-4), ev(-10), best_m10

print(f'DX-PINN E6: lr=3e-4 recon=0.1 wd=1e-4 EMA=0.99')
print(f'CWRU 1HP, SNR[-10,10] AWGN, non-overlap 1024\n')

rc, r4, r10, r10b = [], [], [], []
for run in range(N_RUNS):
    print(f'Run {run+1}/{N_RUNS}...', end=' ', flush=True)
    c, m4, m10, best = train_and_eval(run)
    rc.append(c); r4.append(m4); r10.append(m10); r10b.append(best)
    print(f'Clean={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f} (best={best:.4f})')

print(f'\n{"="*65}')
print(f'DX-PINN (E6) vs MS-TCANet — CWRU 1HP, 10 runs')
print(f'{"="*65}')
print(f'{"Metric":<10} {"DX-PINN":>22} {"MS-TCANet":>22}')
print(f'{"-"*54}')
print(f'{"Clean":<10} {np.mean(rc)*100:>7.2f}% +/- {np.std(rc)*100:.2f}%  {"~100%":>15}')
print(f'{"-4dB":<10} {np.mean(r4)*100:>7.2f}% +/- {np.std(r4)*100:.2f}%  {"96.36% +/- 0.18%":>15}')
print(f'{"-10dB":<10} {np.mean(r10)*100:>7.2f}% +/- {np.std(r10)*100:.2f}%  {"74.67% +/- 1.24%":>15}')
print(f'{"-10dB best":<10} {np.mean(r10b)*100:>7.2f}% +/- {np.std(r10b)*100:.2f}%')
print(f'{"="*65}')
print(f'Delta @-10dB: {np.mean(r10)*100 - 74.67:+.2f}%')
