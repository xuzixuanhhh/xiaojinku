"""域泛化损失"""
import torch.nn as nn


def domain_classification_loss(domain_logits, domain_ids):
    return nn.CrossEntropyLoss()(domain_logits, domain_ids)
