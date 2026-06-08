"""概念瓶颈损失: L_concept + L_EAP"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ccb import EAP_PRIORS


def concept_supervision_loss(c_hat, physics_concepts):
    """L_concept: 概念预测值与物理计算值之间的 MSE"""
    physics_stack = torch.stack([
        physics_concepts["c1"], physics_concepts["c2"],
        physics_concepts["c3"], physics_concepts["c4"],
        physics_concepts["c5"], physics_concepts["c6"]
    ], dim=1)
    return nn.MSELoss()(c_hat, physics_stack)


def eap_loss(c_hat, labels, device):
    """L_EAP: 期望归因先验损失"""
    eap_priors_device = EAP_PRIORS.to(device)
    expected = eap_priors_device[labels]
    excess = F.relu(c_hat - expected)
    return (excess ** 2).mean()


def cac_score(c_hat, labels, device):
    """概念归因一致性: Spearman 相关"""
    from scipy.stats import spearmanr
    eap_priors_device = EAP_PRIORS.to(device)
    actual = c_hat.detach().cpu().numpy()
    expected = eap_priors_device[labels].detach().cpu().numpy()
    corrs = []
    for i in range(len(actual)):
        if actual[i].std() > 1e-6:
            corr, _ = spearmanr(actual[i], expected[i])
            corrs.append(corr)
    return float(sum(corrs) / max(1, len(corrs)))
