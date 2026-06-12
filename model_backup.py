import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *


# ===================== 基础残差块 =====================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class FeatureSoftThreshold(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.threshold = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_scale = x.abs().mean(dim=2)
        threshold = self.threshold(x) * channel_scale
        threshold = threshold.unsqueeze(-1)
        return torch.sign(x) * F.relu(x.abs() - threshold)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shrink = FeatureSoftThreshold(out_channels)

    def forward(self, x):
        return self.shrink(self.block(x))


class RobustSignalPreprocessor(nn.Module):
    def forward(self, x):
        eps = 1e-6
        x = x - x.mean(dim=1, keepdim=True)
        return x / (x.std(dim=1, keepdim=True) + eps)


# ===================== 时域分支编码器：MDCSFormer =====================
class MultiScaleDilatedConvStem(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        branch_dim = hidden_dim // 4
        self.pre = ConvBNAct(1, branch_dim, kernel_size=15, stride=2, dilation=1, dropout=dropout)
        self.branches = nn.ModuleList([
            ConvBNAct(branch_dim, branch_dim, kernel_size=3, stride=1, dilation=1, dropout=dropout),
            ConvBNAct(branch_dim, branch_dim, kernel_size=3, stride=1, dilation=2, dropout=dropout),
            ConvBNAct(branch_dim, branch_dim, kernel_size=3, stride=1, dilation=4, dropout=dropout),
            ConvBNAct(branch_dim, branch_dim, kernel_size=3, stride=1, dilation=8, dropout=dropout),
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(branch_dim, hidden_dim, kernel_size=1),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

    def forward(self, x):
        x = self.pre(x)
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fuse(multi_scale) + self.shortcut(x)


class TimeDomainEncoder(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, num_layers=ENCODER_LAYERS, dropout=DROPOUT_RATE):
        super().__init__()
        self.stem = MultiScaleDilatedConvStem(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers // 2))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        seq = self.stem(x).transpose(1, 2)
        seq = self.transformer(seq)
        seq = self.norm(seq)
        feat = seq.mean(dim=1)
        return seq, feat


# ===================== 频域分支编码器：FFT + 轻量 Transformer/1D-CNN =====================
class FrequencyDomainEncoder(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, num_layers=2, dropout=DROPOUT_RATE):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            ConvBNAct(1, hidden_dim // 2, kernel_size=9, stride=2, dilation=1, dropout=dropout),
            ConvBNAct(hidden_dim // 2, hidden_dim, kernel_size=7, stride=2, dilation=1, dropout=dropout),
            ConvBNAct(hidden_dim, hidden_dim, kernel_size=5, stride=2, dilation=1, dropout=dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(spectrum[:, 1:SAMPLE_LENGTH // 2 + 1])
        noise_floor = amplitude.median(dim=1, keepdim=True).values
        amplitude = torch.log1p(amplitude / (noise_floor + 1e-6))
        amplitude = (amplitude - amplitude.mean(dim=1, keepdim=True)) / (amplitude.std(dim=1, keepdim=True) + 1e-6)
        seq = self.freq_cnn(amplitude.unsqueeze(1)).transpose(1, 2)
        seq = self.transformer(seq)
        seq = self.norm(seq)
        feat = seq.mean(dim=1)
        return seq, feat


# ===================== 交叉注意力融合模块 =====================
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.time_to_freq = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.freq_to_time = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
        )

    def forward(self, time_seq, freq_seq):
        t_attn, _ = self.time_to_freq(time_seq, freq_seq, freq_seq, need_weights=False)
        f_attn, _ = self.freq_to_time(freq_seq, time_seq, time_seq, need_weights=False)
        t_seq = self.norm_t(time_seq + t_attn)
        f_seq = self.norm_f(freq_seq + f_attn)
        t_feat = t_seq.mean(dim=1)
        f_feat = f_seq.mean(dim=1)
        gate = self.gate(torch.cat([t_feat, f_feat], dim=1))
        fused_pair = torch.cat([gate * t_feat, (1.0 - gate) * f_feat], dim=1)
        return self.fusion(fused_pair), t_feat, f_feat


# ===================== PINN 动力学模块（MCK 二阶系统）=====================
class PINNDynamicsModule(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, output_dim=SAMPLE_LENGTH):
        super().__init__()
        self.output_dim = output_dim
        self.mck_hidden = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        self.mck_out = nn.Linear(64, 3)
        self.displacement_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        with torch.no_grad():
            self.mck_out.bias.data[0] = np.log(1.0)
            self.mck_out.bias.data[1] = np.log(5.0)
            self.mck_out.bias.data[2] = np.log(1000.0)

    def predict_displacement(self, shared_feature, t):
        batch_size, seq_len = t.shape
        feature = shared_feature.unsqueeze(1).expand(batch_size, seq_len, shared_feature.shape[1])
        combined = torch.cat([feature, t.unsqueeze(-1)], dim=-1)
        return self.displacement_net(combined.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len)

    def forward(self, shared_feature, t):
        mck_hidden = self.mck_hidden(shared_feature)
        mck_logits = self.mck_out(mck_hidden)
        mck_params = F.softplus(mck_logits)
        m, c, k = mck_params[:, 0:1], mck_params[:, 1:2], mck_params[:, 2:3]
        x_pred = self.predict_displacement(shared_feature, t)
        v_pred = torch.gradient(x_pred, dim=1)[0]
        a_pred_physics = torch.gradient(v_pred, dim=1)[0]
        f_pred = m * a_pred_physics + c * v_pred + k * x_pred
        return {
            "x_pred": x_pred,
            "v_pred": v_pred,
            "a_pred_physics": a_pred_physics,
            "m": m,
            "c": c,
            "k": k,
            "f_pred": f_pred,
        }


# ===================== 预测头组：MCK物理 + 信号重建 =====================
class TimeDomainHead(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, output_dim=SAMPLE_LENGTH):
        super().__init__()
        self.mlp_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.pinn_module = PINNDynamicsModule(hidden_dim, output_dim)
        self.fusion_weight = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, shared_feature, t):
        x_pred_mlp = self.mlp_branch(shared_feature)
        pinn_output = self.pinn_module(shared_feature, t)
        x_pred_pinn = pinn_output["x_pred"]
        a_pred_pinn = pinn_output["a_pred_physics"]
        alpha = self.fusion_weight(shared_feature)
        x_pred = alpha * x_pred_mlp + (1 - alpha) * x_pred_pinn
        return x_pred, a_pred_pinn, pinn_output


class FrequencyDomainHead(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, num_freq_bins=SAMPLE_LENGTH // 2):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.amplitude_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_freq_bins),
            nn.Sigmoid()
        )

    def forward(self, shared_feature):
        return self.amplitude_net(shared_feature)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim=ENCODER_HIDDEN_DIM, classifier_hidden_dim=CLASSIFIER_HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, shared_feature):
        return self.classifier(shared_feature)


# ===================== 强噪声专用升级版 TF-PINN =====================
class TF_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = RobustSignalPreprocessor()
        self.time_encoder = TimeDomainEncoder()
        self.freq_encoder = FrequencyDomainEncoder()
        self.cross_attention_fusion = CrossAttentionFusion()
        self.time_domain_head = TimeDomainHead()
        self.frequency_domain_head = FrequencyDomainHead()
        self.classification_head = ClassificationHead()
        self.num_freq_bins = SAMPLE_LENGTH // 2

    def forward(self, x, t):
        x_raw = x
        x = self.preprocessor(x)
        time_seq, time_feature = self.time_encoder(x)
        freq_seq, freq_feature = self.freq_encoder(x)
        fused_feature, time_feature, freq_feature = self.cross_attention_fusion(time_seq, freq_seq)

        x_pred, a_pred, pinn_output = self.time_domain_head(fused_feature, t)
        cls_pred = self.classification_head(fused_feature)
        pred_amplitude = self.frequency_domain_head(fused_feature)

        spectrum_full = torch.complex(
            pred_amplitude,
            torch.zeros_like(pred_amplitude)
        )
        freq_pred_signal = torch.fft.irfft(spectrum_full, n=SAMPLE_LENGTH, dim=1)

        return {
            "shared_feature": fused_feature,
            "time_feature": time_feature,
            "freq_feature": freq_feature,
            "preprocessed_signal": x,
            "raw_signal": x_raw,
            "x_pred": x_pred,
            "a_pred": a_pred,
            "pred_amplitude": pred_amplitude,
            "freq_pred_signal": freq_pred_signal,
            "cls_pred": cls_pred,
            "v_pred": pinn_output["v_pred"],
            "m": pinn_output["m"],
            "c": pinn_output["c"],
            "k": pinn_output["k"],
        }


if __name__ == "__main__":
    model = TF_PINN().to(DEVICE)
    test_x = torch.randn(2, SAMPLE_LENGTH).to(DEVICE)
    test_t = torch.linspace(0, 1, SAMPLE_LENGTH).repeat(2, 1).to(DEVICE)
    output = model(test_x, test_t)
    print("模型前向传播正常，输出形状：")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
