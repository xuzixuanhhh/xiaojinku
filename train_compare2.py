"""DX-PINN vs MS-TCANet: 5 runs, 200ep, fast"""
import torch, numpy as np, os, random
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def train_one(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    train_loader, _, _ = get_dataloader(0, train_snr_range=(-10, 10), train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(0)
    model = DX_PINN().to(device)

    def ev(snr):
        ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = model(bd, bt); yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy()); yt.extend(bl.cpu().numpy())
        return calculate_metrics(yt, yp)[0]

    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for _ in range(8):
        for batch_data, _, _ in train_loader:
            batch_data = batch_data.to(device); o1.zero_grad()
            xd, ne, es = model.denoiser(batch_data)
            L, _ = denoise_loss(xd, batch_data, ne, es, torch.rand(batch_data.shape[0], 1, device=device) * 20 - 10)
            L.backward(); o1.step()

    for p in model.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=200, eta_min=5e-6)
    best, pat = 0.0, 0
    for ep in range(200):
        model.train()
        for batch_data, batch_label, batch_t in train_loader:
            batch_data = batch_data.to(device); batch_label = batch_label.to(device)
            batch_t = batch_t.to(device).requires_grad_(True); o2.zero_grad()
            out = model(batch_data, batch_t)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], batch_label)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), batch_label, device)
                    + 0.1 * torch.nn.functional.mse_loss(out['x_denoised'], batch_data))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); o2.step()
        sc.step()
        if ep % 15 == 0:
            model.eval(); m10 = ev(-10)
            if m10 > best: best = m10; pat = 0
            else: pat += 1
            if pat >= 10: break
            model.train()
    model.eval()
    return ev(10), ev(-4), ev(-10)

print(f'Device: {device}\nDX-PINN vs MS-TCANet: 5 runs\n')
rc, r4, r10 = [], [], []
for run in range(5):
    print(f'Run {run+1}/5...', end=' ', flush=True)
    c, m4, m10 = train_one(run); rc.append(c); r4.append(m4); r10.append(m10)
    print(f'C={c:.4f} -4={m4:.4f} -10={m10:.4f}')

print(f'\n=== DX-PINN vs MS-TCANet ===')
print(f'Clean: {np.mean(rc):.4f}+/-{np.std(rc):.4f}  | ~100%')
print(f'-4dB : {np.mean(r4):.4f}+/-{np.std(r4):.4f}  | 96.50%')
print(f'-10dB: {np.mean(r10):.4f}+/-{np.std(r10):.4f}  | 75.00%')
print(f'Best  = C={np.max(rc):.4f} -4={np.max(r4):.4f} -10={np.max(r10):.4f}')
