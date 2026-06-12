"""Test 100-epoch configs: short SUD pretrain + warmup + high LR + EMA"""
import torch, numpy as np, random, copy
from config import *
from data_loader import get_dataloader, BearingDataset, load_dataset
from model.dx_pinn import DX_PINN
from loss.denoise_loss import denoise_loss
from loss.concept_loss import eap_loss
from utils import calculate_metrics
from torch.utils.data import DataLoader

device = DEVICE

def test(name, pretrain_ep, warmup_ep, lr_max, ema_decay, seeds=[42, 123, 456]):
    results = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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

        # P1: short SUD pretrain
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

        # P2: joint training with warmup + EMA
        for p in m.parameters(): p.requires_grad = True
        o2 = torch.optim.AdamW(m.parameters(), lr=lr_max, weight_decay=1e-4)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=100 - warmup_ep, eta_min=1e-5)
        ema = copy.deepcopy(m)
        for p in ema.parameters(): p.requires_grad = False
        best_m10 = 0.0

        for ep in range(100):
            m.train()
            if ep < warmup_ep:
                for pg in o2.param_groups:
                    pg['lr'] = lr_max * (ep + 1) / warmup_ep
            for bd, bl, bt in tl:
                bd, bl, bt = bd.to(device), bl.to(device), bt.to(device).requires_grad_(True)
                o2.zero_grad()
                out = m(bd, bt)
                loss = (torch.nn.functional.cross_entropy(out['cls_pred'], bl)
                        + 3.0 * eap_loss(torch.sigmoid(out['c_hat']), bl, device)
                        + 0.1 * torch.nn.functional.mse_loss(out['x_denoised'], bd))
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o2.step()
                with torch.no_grad():
                    a = 1 - ema_decay
                    for pe, p in zip(ema.parameters(), m.parameters()):
                        pe.data.mul_(ema_decay).add_(p.data, alpha=a)
            if ep >= warmup_ep:
                sc.step()
            if ep % 10 == 0:
                ema.eval(); m10 = ev(-10)
                if m10 > best_m10: best_m10 = m10
                m.train()
        ema.eval()
        results.append((ev(-10), ev(-4), best_m10))
    return results

configs = [
    ('C1_p3_w0_lr1e-3',      3, 0, 1e-3,   0.995),
    ('C2_p3_w4_lr1e-3',      3, 4, 1e-3,   0.995),
    ('C3_p3_w4_lr1.5e-3',    3, 4, 1.5e-3, 0.995),
    ('C4_p3_w4_lr1e-3_099',  3, 4, 1e-3,   0.99),
    ('C5_p5_w5_lr1e-3',      5, 5, 1e-3,   0.995),
]

print('100-epoch test: short pretrain + warmup + EMA')
for name, pe, we, lr, ema in configs:
    res = test(name, pe, we, lr, ema)
    m10s = [r[0] for r in res]; m4s = [r[1] for r in res]; bests = [r[2] for r in res]
    vals_str = ', '.join([format(v, '.3f') for v in m10s])
    best_max = max(bests)
    print('{}: -10dB mean={:.4f} [{}] -4dB={:.4f} best={:.4f}'.format(
        name, np.mean(m10s), vals_str, np.mean(m4s), best_max))

print()
print('Ref: 200ep E2=78.10%, MS-TCANet=74.67%')
