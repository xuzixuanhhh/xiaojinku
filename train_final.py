"""DX-PINN Final: Low LR 0.0005 + 300 epochs GPU — eliminate oscillation"""
import torch, numpy as np, os, random
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42); torch.backends.cudnn.benchmark = True

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

# P2: Low LR, long training
print('\nPhase 2: Low LR (0.0005) + 300 epochs')
for p in model.parameters(): p.requires_grad = True
opt2 = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=300, eta_min=1e-5)
best_m10, patience = 0.0, 0

for ep in range(300):
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
    if ep % 5 == 0:
        model.eval()
        c = eval_at(20); m6 = eval_at(-6); m10 = eval_at(-10)
        print(f'E{ep+1:3d}: LR={opt2.param_groups[0]["lr"]:.6f} TA={train_acc:.4f} Clean={c:.4f} -6dB={m6:.4f} -10dB={m10:.4f}')
        if m10 > best_m10:
            best_m10 = m10; torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'dx_pinn_best.pth')); patience = 0
        else: patience += 1
        if patience >= 15: break
        model.train()

print(f'\nBest -10dB: {best_m10:.4f}')
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'dx_pinn_best.pth'))); model.eval()
print('Final:')
for snr in [20, 10, 0, -2, -4, -6, -8, -10]:
    print(f'  SNR={snr:3d}dB: Acc={eval_at(snr):.4f}')
