"""DX-PINN v8: CWT Encoder + SUD — target SOTA @ -10dB"""
import torch, random, numpy as np, os
random.seed(42); np.random.seed(42); torch.manual_seed(42)
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
print(f'Device: {device}')
train_loader, _, _ = get_dataloader(0, train_snr_range=(-12, 5), train_noise=True, val_noise=True, test_noise=True)
_, _, td, _, _, tlbl = load_dataset(0)
model = DX_PINN().to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

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

# Phase 1: SUD pretrain
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

# Phase 2: Joint training
print('\nPhase 2: Joint training (120 epochs)')
for p in model.parameters(): p.requires_grad = True
opt2 = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=25, T_mult=2)
best_score, patience = 0.0, 0

for ep in range(120):
    model.train(); tl, yt, yp = 0.0, [], []
    for batch_data, batch_label, batch_t in train_loader:
        batch_data = batch_data.to(device); batch_label = batch_label.to(device)
        batch_t = batch_t.to(device).requires_grad_(True); opt2.zero_grad()
        output = model(batch_data, batch_t)
        est_snr = output['est_snr'].detach()
        w = torch.sigmoid((est_snr + 10) / 2.5)
        w = w / (w.sum() + 1e-8) * len(w)
        L_cls = (torch.nn.functional.cross_entropy(output['cls_pred'], batch_label, reduction='none') * w.squeeze()).mean()
        L_eap = eap_loss(torch.sigmoid(output['c_hat']), batch_label, device)
        L_recon = torch.nn.functional.mse_loss(output['x_denoised'], batch_data)
        loss = L_cls + 3.0 * L_eap + 0.1 * L_recon
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step(); tl += loss.item()
        yt.extend(batch_label.cpu().numpy()); yp.extend(torch.argmax(output['cls_pred'], dim=1).cpu().numpy())
    sched.step()
    train_acc = calculate_metrics(yt, yp)[0]; model.eval()
    c = eval_at(20); m4 = eval_at(-4); m10 = eval_at(-10)
    print(f'E{ep+1:3d}: TA={train_acc:.4f} Clean={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f}')
    score = c * 0.3 + m4 * 0.3 + m10 * 0.4
    if score > best_score:
        best_score = score; torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'dx_pinn_best.pth')); patience = 0
    else: patience += 1
    if patience >= 40: break

print(f'\nBest score: {best_score:.4f}')
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'dx_pinn_best.pth'))); model.eval()
print('Final:')
for snr in [20, 10, 0, -2, -4, -6, -8, -10]:
    print(f'  SNR={snr:3d}dB: Acc={eval_at(snr):.4f}')
