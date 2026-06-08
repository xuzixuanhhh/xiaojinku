"""Multi-View Manifold Contrastive Domain Generalization"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ENCODER_HIDDEN_DIM


class MVMCDG(nn.Module):
    """多视角流形对比域泛化

    输入: time_feature, freq_feature, c_hat, labels, domain_ids
    输出: L_hsic, L_cc, L_proto
    """

    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, temperature=0.1, k_nn=5):
        super().__init__()
        self.temperature = temperature
        self.k_nn = k_nn
        self.view_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.prototypes = nn.Parameter(torch.randn(10, 6) * 0.1)
        self.proto_momentum = 0.99

    def knn_graph_aggregate(self, features, k=5):
        B = features.shape[0]
        if B < k + 1:
            return features
        sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        _, idx = sim.topk(k + 1, dim=1)
        idx = idx[:, 1:]
        neighbors = features[idx]
        return 0.5 * features + 0.5 * neighbors.mean(dim=1)

    def compute_hsic(self, x, y):
        B = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        Kx = torch.mm(x, x.t())
        Ky = torch.mm(y, y.t())
        H = torch.eye(B, device=x.device) - 1.0 / B
        HKxH = torch.mm(torch.mm(H, Kx), H)
        hsic = torch.trace(torch.mm(HKxH, Ky)) / ((B - 1) ** 2)
        return torch.abs(hsic)

    def supervised_contrastive_loss(self, c_hat, labels):
        B = c_hat.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=c_hat.device)
        c_norm = F.normalize(c_hat, dim=1)
        sim = torch.mm(c_norm, c_norm.t()) / self.temperature
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask.fill_diagonal_(False)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        return -torch.log(pos_sum / (all_sum + 1e-8)).mean()

    def prototype_loss(self, c_hat, labels, margin=0.5):
        B = c_hat.shape[0]
        proto = self.prototypes[labels]
        L_pos = F.mse_loss(c_hat, proto)
        L_neg = torch.tensor(0.0, device=c_hat.device)
        for i in range(B):
            neg_mask = torch.ones(10, device=c_hat.device).bool()
            neg_mask[labels[i]] = False
            neg_protos = self.prototypes[neg_mask]
            dists = ((c_hat[i:i + 1] - neg_protos) ** 2).mean(dim=1)
            L_neg += F.relu(margin - dists).mean()
        return L_pos + 0.1 * (L_neg / max(1, B))

    def update_prototypes(self, c_hat, labels):
        with torch.no_grad():
            for cls in range(10):
                mask = labels == cls
                if mask.sum() > 0:
                    cls_mean = c_hat[mask].mean(dim=0)
                    self.prototypes[cls] = (
                        self.proto_momentum * self.prototypes[cls]
                        + (1 - self.proto_momentum) * cls_mean
                    )

    def forward(self, time_feature, freq_feature, c_hat, labels, domain_ids):
        fusion_in = torch.cat([time_feature, freq_feature], dim=1)
        fused = self.view_fusion(fusion_in)
        fused_agg = self.knn_graph_aggregate(fused, self.k_nn)
        domain_onehot = F.one_hot(domain_ids, num_classes=4).float()
        L_hsic = self.compute_hsic(fused_agg, domain_onehot)
        L_cc = self.supervised_contrastive_loss(c_hat, labels)
        L_proto = self.prototype_loss(c_hat, labels)
        return {"L_hsic": L_hsic, "L_cc": L_cc, "L_proto": L_proto}
