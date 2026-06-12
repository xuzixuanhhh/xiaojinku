"""DX-PINN vs MS-TCANet under matched conditions:
- CWRU 1HP, non-overlap 1024, 10 classes
- AWGN SNR [-10,10], dynamic injection (exact paper protocol)
- lr=0.0005 (DX-PINN proven recipe), 200 epochs
"""
import torch, numpy as np, random
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
TRAIN_SNR = (-10, 10)  # exact match to paper

train_loader, _, _ = get_dataloader(1, train_snr_range=TRAIN_SNR,
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

print('P1: SUD pretrain (8 epochs)')
for p in model.parameters(): p.requires_grad = False
for p in model.denoiser.parameters(): p.requires_grad = True
o1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
for _ in range(8):
    for bd, _, _ in train_loader:
        bd = bd.to(device); o1.zero_grad()
        xd, ne, es = model.denoiser(bd)
        ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10  # [-10, 10]
        L, _ = denoise_loss(xd, bd, ne, es, ts)
        L.backward(); o1.step()

print('P2: lr=5e-4, CosineAnnealing, 200 epochs')
for p in model.parameters(): p.requires_grad = True
o2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=200, eta_min=1e-5)
best_m10, best_ep, patience = 0.0, 0, 0

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
    sc.step()
    if ep % 10 == 0:
        model.eval(); m10 = ev(-10)
        if m10 > best_m10: best_m10 = m10; best_ep = ep+1; patience = 0
        else: patience += 1
        print(f'  E{ep+1:3d}: -10dB={m10:.4f} best={best_m10:.4f}@{best_ep}')
        if patience >= 15: break
        model.train()

model.eval()
print(f'\n=== DX-PINN vs MS-TCANet (CWRU 1HP, SNR[-10,10] AWGN) ===')
for snr in [10, -4, -10]:
    acc = ev(snr)
    ref = {10: '~100%', -4: '96.36%', -10: '74.67%'}
    print(f'  SNR={snr:3d}dB: DX={acc*100:.2f}% | Paper={ref[snr]}')
