"""DX-PINN 三阶段训练"""
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *
from data_loader import get_dataloader, get_dg_dataloader
from model.dx_pinn import DX_PINN
from model.ccb import compute_physics_concepts
from loss import total_loss as old_total_loss
from loss.denoise_loss import denoise_loss
from loss.concept_loss import concept_supervision_loss, eap_loss
from loss.dg_loss import domain_classification_loss
from utils import calculate_metrics


def evaluate_model(model, loader, device):
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch_data, batch_label, batch_t in loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_t = batch_t.to(device).requires_grad_(True)
            output = model(batch_data, batch_t)
            pred_label = torch.argmax(output["cls_pred"], dim=1)
            all_y_true.extend(batch_label.cpu().numpy())
            all_y_pred.extend(pred_label.cpu().numpy())
    acc, _, _, _ = calculate_metrics(all_y_true, all_y_pred)
    return acc


def train_phase1(model, train_loader, device, epochs=5):
    """预训练 SUD 去噪模块"""
    print("=" * 50 + "\nPhase 1: 预训练 Stockwell 去噪模块")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.denoiser.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.denoiser.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total = 0.0
        for batch_data, batch_label, batch_t in tqdm(train_loader, desc=f"P1 E{epoch + 1}"):
            batch_data = batch_data.to(DEVICE)
            opt.zero_grad()
            x_denoised, noise_est, est_snr = model.denoiser(batch_data)
            true_snr = torch.full((batch_data.shape[0], 1), -5.0, device=DEVICE)
            L, _ = denoise_loss(x_denoised, batch_data, noise_est, est_snr, true_snr)
            L.backward()
            opt.step()
            total += L.item()
        print(f"  Loss: {total / len(train_loader):.6f}")


def train_phase2(model, train_loader, val_loader, device, epochs=EPOCHS):
    """联合训练所有模块"""
    print("=" * 50 + "\nPhase 2: 联合训练")
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LEARNING_RATE, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos')
    best_val, pat = 0.0, 0
    log = []

    for epoch in range(epochs):
        model.train()
        tl, yt, yp = 0.0, [], []
        for batch_data, batch_label, batch_t in tqdm(train_loader, desc=f"P2 E{epoch + 1}"):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_t = batch_t.to(device).requires_grad_(True)
            opt.zero_grad()
            output = model(batch_data, batch_t)

            L_cls = torch.nn.functional.cross_entropy(output["cls_pred"], batch_label)
            L_eap = eap_loss(output["c_hat"], batch_label, device)

            loss = L_cls + EAP_LAMBDA * L_eap
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            yt.extend(batch_label.cpu().numpy())
            yp.extend(torch.argmax(output["cls_pred"], dim=1).cpu().numpy())

        train_acc, _, _, _ = calculate_metrics(yt, yp)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"  Train Loss={tl / len(train_loader):.4f} Acc={train_acc:.4f} Val Acc={val_acc:.4f}")
        log.append({"epoch": epoch + 1, "train_loss": tl / len(train_loader),
                     "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "dx_pinn_best.pth"))
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    pd.DataFrame(log).to_csv(os.path.join(SAVE_PATH, "phase2_log.csv"), index=False)
    return best_val


def train_phase3(model, device, epochs=20):
    """域泛化微调"""
    print("=" * 50 + "\nPhase 3: 域泛化微调")
    dg_loader, _ = get_dg_dataloader(DG_TRAIN_CONDITIONS, TRAIN_SNR_RANGE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        alpha = min(1.0, epoch / GRL_WARMUP_EPOCHS)
        model.set_grl_alpha(alpha)
        tl, yt, yp = 0.0, [], []
        for batch_data, batch_label, batch_t, domain_ids in tqdm(dg_loader, desc=f"P3 E{epoch + 1}"):
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_t = batch_t.to(device).requires_grad_(True)
            domain_ids = domain_ids.to(device)
            opt.zero_grad()
            output = model(batch_data, batch_t, labels=batch_label, domain_ids=domain_ids, return_dg=True)

            L_cls = torch.nn.functional.cross_entropy(output["cls_pred"], batch_label)
            L_domain = domain_classification_loss(output["domain_logits"], domain_ids)
            physics = compute_physics_concepts(output["x_denoised"])
            L_concept = concept_supervision_loss(output["c_hat"], physics)
            L_eap = eap_loss(output["c_hat"], batch_label, device)
            dg = output["dg_output"]

            loss = 8.0 * L_cls + L_domain + CONCEPT_LAMBDA * L_concept + EAP_LAMBDA * L_eap \
                + HSIC_LAMBDA * dg["L_hsic"] + CC_LAMBDA * dg["L_cc"] + PROTO_LAMBDA * dg["L_proto"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            model.mvm_cdg.update_prototypes(output["c_hat"], batch_label)
            yt.extend(batch_label.cpu().numpy())
            yp.extend(torch.argmax(output["cls_pred"], dim=1).cpu().numpy())
        acc, _, _, _ = calculate_metrics(yt, yp)
        print(f"  Loss={tl / len(dg_loader):.4f} Acc={acc:.4f} GRL={alpha:.2f}")
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "dx_pinn_final.pth"))


def train_dx_pinn():
    print("加载数据...")
    train_loader, val_loader, _ = get_dataloader(
        TRAIN_WORK_CONDITION, train_snr_range=TRAIN_SNR_RANGE,
        train_noise=True, val_noise=True, test_noise=True)
    print("初始化 DX-PINN...")
    model = DX_PINN().to(DEVICE)
    train_phase1(model, train_loader, DEVICE)
    train_phase2(model, train_loader, val_loader, DEVICE)
    train_phase3(model, DEVICE)
    print("训练完成! 模型保存至 results/")
    return model


if __name__ == "__main__":
    train_dx_pinn()
