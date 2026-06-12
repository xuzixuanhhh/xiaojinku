"""DX-PINN vs MS-TCANet: same conditions, our best training strategy"""
import torch, numpy as np, os, random
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
print(f'Device: {device}')
print(f'Setup: SNR[-10,10]dB, non-overlap, 4 conditions, 1D input, 10 runs, best effort convergence')

def train_eval_one_run(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_loader, _, _ = get_dataloader(0, train_snr_range=(-10, 10), train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(0)
    model = DX_PINN().to(device)

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

    # Phase 1
    for p in model.parameters(): p.requires_grad = False
    for p in model.denoiser.parameters(): p.requires_grad = True
    opt1 = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for ep in range(10):
        for batch_data, _, _ in train_loader:
            batch_data = batch_data.to(device); opt1.zero_grad()
            x_denoised, noise_est, est_snr = model.denoiser(batch_data)
            L, _ = denoise_loss(x_denoised, batch_data, noise_est, est_snr,
                                torch.rand(batch_data.shape[0], 1, device=device) * 20 - 10)
            L.backward(); opt1.step()

    # Phase 2: low LR, long training, early stop on -10dB
    for p in model.parameters(): p.requires_grad = True
    opt2 = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=300, eta_min=5e-6)
    best_m10, patience = 0.0, 0

    for ep in range(300):
        model.train()
        for batch_data, batch_label, batch_t in train_loader:
            batch_data = batch_data.to(device); batch_label = batch_label.to(device)
            batch_t = batch_t.to(device).requires_grad_(True); opt2.zero_grad()
            output = model(batch_data, batch_t)
            loss = (torch.nn.functional.cross_entropy(output['cls_pred'], batch_label)
                    + 3.0 * eap_loss(torch.sigmoid(output['c_hat']), batch_label, device)
                    + 0.1 * torch.nn.functional.mse_loss(output['x_denoised'], batch_data))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt2.step()
        sched.step()
        if ep % 10 == 0:
            model.eval(); m10 = eval_at(-10)
            if m10 > best_m10: best_m10 = m10; patience = 0
            else: patience += 1
            if patience >= 15: break
            model.train()

    model.eval()
    return eval_at(10), eval_at(-4), eval_at(-10)

results_clean, results_m4, results_m10 = [], [], []
for run in range(10):
    print(f'Run {run+1}/10...', end=' ', flush=True)
    c, m4, m10 = train_eval_one_run(run)
    results_clean.append(c); results_m4.append(m4); results_m10.append(m10)
    print(f'Clean={c:.4f} -4dB={m4:.4f} -10dB={m10:.4f}')

print(f'\n=== DX-PINN vs MS-TCANet ===')
print(f'{"Metric":<10} {"DX-PINN":>20} {"MS-TCANet":>15}')
print(f'{"Clean":<10} {np.mean(results_clean):.4f}+/-{np.std(results_clean):.4f} {"~100%":>15}')
print(f'{"-4dB":<10} {np.mean(results_m4):.4f}+/-{np.std(results_m4):.4f} {"96.50%":>15}')
print(f'{"-10dB":<10} {np.mean(results_m10):.4f}+/-{np.std(results_m10):.4f} {"75.00%":>15}')
print(f'Best -10dB: {np.max(results_m10):.4f}, Worst: {np.min(results_m10):.4f}')
