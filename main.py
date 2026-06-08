"""DX-PINN 完整入口: 训练 + 实验"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(42)
    print("=" * 60)
    print("DX-PINN: Deep eXplainable Physics-Informed Network")
    print("=" * 60)

    from train import train_dx_pinn
    model = train_dx_pinn()

    from test import run_all_experiments
    run_all_experiments()

    print("=" * 60 + "\n完成!")
