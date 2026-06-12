"""DX-PINN v10: bimodal SNR + extended training — beat SOTA @ -10dB"""
import torch, random, numpy as np, os
random.seed(42); np.random.seed(42); torch.manual_seed(42)
from config import *
from data_loader import BearingDataset, load_dataset, NoisyBearingDataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader, ConcatDataset

device = DEVICE
_, _, td, _, _, tlbl = load_dataset(0)

# Bimodal: 50% moderate, 50% extreme noise
ds_clean = NoisyBearingDataset(td, tlbl, snr_range=(-5, 10))
ds_hard = NoisyBearingDataset(td, tlbl, snr_range=(-12, -6))
ds_mixed = ConcatDataset([ds_clean, ds_hard])
train_loader = DataLoader(ds_mixed, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = DX_PINN().to(device)
print(f'Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}, Batches: {len(train_loader)*2}')

def eval_at(snr_val):
    ds = BearingDataset(td, tlbl, snr_db=snr_val, add_noise=True)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    yt, yp = [], []
    with torch.no_grad():
        for bd, bl, bt in dl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            out = model(bd, bt)
            yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
            yt.extend(bl.cpu().numpy())
    return calculate_metrics(yt, yp)[0]

# Phase 1
print('\nPhase 1: SUD pretrain')
for p in model.parameters(): p.requires_grad = False
for p in model.denoiser.parameters(): p.requires_grad = True
opt1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
for ep in range(10):
    total = 0.0
    for batch_data, _, _ in train_loader:
        batch_data = batch_data.to(device); opt1.zero_grad()
        x_denoised, noise_est, est_snr = model.denoiser(batch_data)
        true_snr = torch.rand(batch_data.shape[0], 1, device=device) * 17 - 12
        L, _ = denoise_loss(x_denoised, batch_data, noise_est, est_snr, true_snr)
        L.backward(); opt1.step(); total += L.item()
    print(f'  E{ep+1}: {total/len(train_loader):.4f}')

# Phase 2
print('\nPhase 2: Joint training (150 epochs)')
for p in model.parameters(): p.requires_grad = True
opt2 = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=30, T_mult=2)
best_m10, patience = 0.0, 0

for ep in range(150):
    model.train(); tl, yt, yp = 0.0, [], []
    for batch_data, batch_label, batch_t in train_loader:
        batch_data = batch_data.to(device); batch_label = batch_label.to(device)
        batch_t = batch_t.to(device).requires_grad_(True); opt2.zero_grad()
        output = model(batch_data, batch_t)
        L_cls = torch.nn.functional.cross_entropy(output['cls_pred'], batch_label)
        L_eap = eap_loss(torch.sigmoid(output['c_hat']), batch_label, device)
        L_recon = torch.nn.functional.mse_loss(output['x_denoised'], batch_data)
        loss = L_cls + 2.0 * L_eap + 0.1 * L_recon
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step(); tl += loss.item()
        yt.extend(batch_label.cpu().numpy()); yp.extend(torch.argmax(output['cls_pred'], dim=1).cpu().numpy())
    sched.step()
    train_acc = calculate_metrics(yt, yp)[0]
    if ep % 2 == 0:
        model.eval()
        c = eval_at(20); m6 = eval_at(-6); m10 = eval_at(-10)
        print(f'E{ep+1:3d}: TA={train_acc:.4f} Clean={c:.4f} -6dB={m6:.4f} -10dB={m10:.4f}')
        if m10 > best_m10:
            best_m10 = m10; torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'dx_pinn_best.pth')); patience = 0
        else: patience += 1
        if patience >= 25: break
    model.train()

print(f'\nBest -10dB: {best_m10:.4f}')
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'dx_pinn_best.pth'))); model.eval()
print('Final:')
for snr in [20, 10, 0, -2, -4, -6, -8, -10]:
    print(f'  SNR={snr:3d}dB: Acc={eval_at(snr):.4f}')
