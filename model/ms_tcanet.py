"""MS-TCANet: Exact reproduction from Zhang et al. (Sensors, 2026)
Paper: CWRU 1HP, -10dB = 74.67% +/- 1.24%, -4dB = 96.36% +/- 0.18%
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PACA(nn.Module):
    """Peak-Aware Coordinate Attention: dual-pooling (avg+max) along H,W."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.gn = nn.GroupNorm(1, hidden)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        zh = self.pool_h_avg(x) + self.pool_h_max(x)
        zw = self.pool_w_avg(x) + self.pool_w_max(x)
        zw_t = zw.permute(0, 1, 3, 2)
        f = torch.cat([zh, zw_t], dim=2)
        f = F.silu(self.gn(self.conv1(f)))
        fh, fw = f[:, :, :H, :], f[:, :, H:, :].permute(0, 1, 3, 2)
        ah = torch.sigmoid(self.conv_h(fh))
        aw = torch.sigmoid(self.conv_w(fw))
        return x + x * ah * aw


class RobustMSBlock(nn.Module):
    """Multi-scale DWConv (3x3 + 7x7) + MLP + PACA + residual."""

    def __init__(self, ch):
        super().__init__()
        self.dw3 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.dw7 = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.gn_ms = nn.GroupNorm(1, ch)
        hidden = ch * 4
        self.gn_mlp = nn.GroupNorm(1, ch)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.paca = PACA(ch)

    def forward(self, x):
        r = x
        x = (self.dw3(x) + self.dw7(x)) / 2.0
        x = self.gn_ms(x)
        x_mlp = self.fc2(F.gelu(self.fc1(self.gn_mlp(x))))
        x = x + x_mlp
        x = self.paca(x)
        return r + x


class MS_TCANet(nn.Module):
    """MS-TCANet: Stem + 3 Stages [64,128,256] + FC."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.GroupNorm(1, 64), nn.GELU())
        # Stage 1
        self.s1a = RobustMSBlock(64); self.s1b = RobustMSBlock(64)
        self.ds1 = nn.Sequential(nn.Conv2d(64, 128, 2, stride=2),
                                 nn.GroupNorm(1, 128), nn.GELU())
        # Stage 2
        self.s2a = RobustMSBlock(128); self.s2b = RobustMSBlock(128)
        self.ds2 = nn.Sequential(nn.Conv2d(128, 256, 2, stride=2),
                                 nn.GroupNorm(1, 256), nn.GELU())
        # Stage 3
        self.s3a = RobustMSBlock(256); self.s3b = RobustMSBlock(256)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.s1a(x); x = self.s1b(x); x = self.ds1(x)
        x = self.s2a(x); x = self.s2b(x); x = self.ds2(x)
        x = self.s3a(x); x = self.s3b(x)
        return self.head(x)
