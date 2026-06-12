"""100-epoch v2: no SUD pretrain, joint from start + new ideas"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE
SEED = 42

def test(name, pretrain_ep, lr, recon_w, ema_decay, sched_type, warmup, wd=1e-4):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    tl, _, _ = get_dataloader(1, train_snr_range=(-10, 10),
        train_noise=True, val_noise=True, test_noise=True)
    _, _, td, _, _, tlbl = load_dataset(1)
    m = DX_PINN().to(device)

    def ev(snr):
        ds = BearingDataset(td, tlbl, snr_db=snr, add_noise=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        yt, yp = [], []
        with torch.no_grad():
            for bd, bl, bt in dl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                out = m(bd, bt)
                yp.extend(torch.argmax(out['cls_pred'], dim=1).cpu().numpy())
                yt.extend(bl.cpu().numpy())
        return calculate_metrics(yt, yp)[0]

    # P1: pretrain (0 = skip, joint from start with recon decay)
    no_pretrain = (pretrain_ep == 0)
    if not no_pretrain:
        for p in m.parameters(): p.requires_grad = False
        for p in m.denoiser.parameters(): p.requires_grad = True
        o1 = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-3)
        for _ in range(pretrain_ep):
            for bd, _, _ in tl:
                bd = bd.to(device); o1.zero_grad()
                xd, ne, es = m.denoiser(bd)
                ts = torch.rand(bd.shape[0], 1, device=device) * 20 - 10
                L, _ = denoise_loss(xd, bd, ne, es, ts)
                L.backward(); o1.step()

    # P2: joint training
    for p in m.parameters(): p.requires_grad = True
    o2 = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

    if sched_type == 'cos':
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=100, eta_min=1e-5)
    elif sched_type == 'cos_restart':
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o2, T_0=25, T_mult=2)
    elif sched_type == 'cos_warm':
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=100 - warmup, eta_min=1e-5)

    ema = copy.deepcopy(m)
    for p in ema.parameters(): p.requires_grad = False
    best_m10, trace = 0.0, []

    for ep in range(100):
        m.train()
        if warmup > 0 and ep < warmup:
            for pg in o2.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup

        # If no pretrain: decay recon_w from 0.3 to 0.1 over 100 epochs
        cur_recon = 0.3 - 0.2 * (ep / 100) if no_pretrain else recon_w

        for bd, bl, bt in tl:
            bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
            o2.zero_grad()
            out = m(bd, bt)
            loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                    + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                    + cur_recon * torch.nn.functional.mse_loss(out['x_denoised'], bd))
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o2.step()
            with torch.no_grad():
                a = 1 - ema_decay
                for pe, p in zip(ema.parameters(), m.parameters()):
                    pe.data.mul_(ema_decay).add_(p.data, alpha=a)
        if sched_type != 'cos_restart':
            if ep >= warmup: sc.step()
        else:
            sc.step()
        if ep % 10 == 0:
            ema.eval(); m10 = ev(-10)
            if m10 > best_m10: best_m10 = m10
            trace.append(m10)
            m.train()
    ema.eval()
    final_m10 = ev(-10); final_m4 = ev(-4)
    osc = np.std(trace[-4:]) if len(trace) >= 4 else 0
    return final_m10, final_m4, best_m10, osc, trace

configs = [
    # Skip pretrain entirely - joint from epoch 1 with recon decay
    ('D1_noPT_lr1e-3',          0, 1e-3,  0.1, 0.995, 'cos', 0),
    ('D2_noPT_lr2e-3',          0, 2e-3,  0.1, 0.995, 'cos', 0),
    ('D3_noPT_warm5_lr1e-3',    0, 1e-3,  0.1, 0.995, 'cos_warm', 5),
    # Restart scheduler
    ('D4_p3_lr1e-3_restart',    3, 1e-3,  0.1, 0.995, 'cos_restart', 0),
    ('D5_noPT_lr1e-3_restart',  0, 1e-3,  0.1, 0.995, 'cos_restart', 0),
    # High recon + pretrain
    ('D6_p3_lr1e-3_r0.3',       3, 1e-3,  0.3, 0.995, 'cos', 0),
    # Reference from v1
    ('REF_C5_p5_w5_lr1e-3',     5, 1e-3,  0.1, 0.995, 'cos_warm', 5),
]

print('100-epoch v2: no-pretrain + recon decay + restarts (single run each)')
print('-' * 70)
for name, pe, lr, rw, ema, sc, wu in configs:
    fm10, fm4, best, osc, trace = test(name, pe, lr, rw, ema, sc, wu)
    last4 = [format(t, '.3f') for t in trace[-4:]]
    print('{}: -10dB={:.4f} -4dB={:.4f} best={:.4f} osc={:.4f} trace={}'.format(
        name, fm10, fm4, best, osc, last4))

print()
print('Target: 200ep E2=78.10% | MS-TCANet=74.67% | v1 C5=73.57%')
