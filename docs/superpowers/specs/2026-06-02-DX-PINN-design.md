# DX-PINN: Deep eXplainable Physics-Informed Network for Noise-Robust Bearing Fault Diagnosis

## Overview

DX-PINN 是一个面向轴承故障诊断的深度可解释物理信息神经网络，聚焦三个核心能力：
- **XAI（内建可解释性）**：因果概念瓶颈 + 期望归因先验损失
- **强噪声鲁棒（-10dB > 90%）**：Stockwell 展开式可学习去噪
- **跨工况域泛化**：多视角流形对比学习 + HSIC 特征解耦

基线在现有 TF-PINN 基础上迭代，吸收 2024-2026 年 9 篇顶刊论文的创新点进行深度融合。

---

## Architecture

```
                          ┌──────────────────────────────────────┐
    x(t) + noise ──────▶ │  ① SUD: Stockwell Unrolled Denoiser   │
                          │  · Sinc 卷积核匹配故障频带             │
                          │  · 3 层 ISTA 算法展开 (逐层可解释)     │
                          │  · 自适应剪枝 (SNR < -6dB)            │
                          └────────────────┬─────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
              时域 MDCSFormer                           频域 FFT+CNN+TF
              (保留,增强)                               (保留,增强)
                      │                                         │
                      └──────────────┬──────────────────────────┘
                                     ▼
                           交叉注意力融合 (保留)
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
       ┌──────────────────────────┐   ┌──────────────────────────┐
       │ ② CCB: 因果概念瓶颈       │   │ ③ MVM-CDG: 多视角流形对比  │
       │ · 因果解耦: 故障/工况分离  │   │ · 时频域动态 K-NN 图       │
       │ · 6 物理概念              │   │ · HSIC 域-类特征解耦       │
       │ · EAP 期望归因先验损失     │   │ · 概念空间监督对比损失     │
       └──────────┬───────────────┘   └──────────┬───────────────┘
                  │                              │
                  ▼                              ▼
           概念→分类器                      域分类器(对抗)
                  │
                  ▼
           ┌──────────────────────────┐
           │ XAI 输出:                 │
           │ · 概念激活雷达图          │
           │ · 归因先验一致性分数       │
           │ · 故障/工况解耦可视化      │
           │ · 诊断决策路径             │
           └──────────────────────────┘
```

---

## Module 1: SUD (Stockwell Unrolled Denoiser)

### 灵感来源
- PFISTA-Net (Zhong et al., Neurocomputing 2026): Sinc 函数物理卷积核 + 算法展开
- SAFATN (EAAI 2025): Stockwell 变换权重初始化
- LSR-Net (Lee et al., arXiv 2026): 自适应剪枝

### 设计
1. **Sinc 卷积核初始化**：第一层 Conv1d 权重用 Sinc 函数族初始化，中心频率匹配 CWRU 轴承故障特征频带 (BPFI/BPFO/BSF 及其倍频)
2. **ISTA 算法展开**：3 层迭代软阈值去噪，每层公式:
   ```
   z^(k+1) = soft(z^(k) - alpha * W^T(W * z^(k) - y), lambda^(k))
   ```
   其中 soft 阈值 lambda^(k) 是可学习参数，每层独立
3. **自适应深度**：当估计 SNR < -6dB 时激活全部 3 层；噪声较轻时只用 1-2 层
4. **输出**：去噪信号 + 噪声估计（用于损失函数中的噪声感知项）

### 参数量
- Sinc conv kernels: 64 x 1 x 15
- ISTA 参数: alpha, lambda per layer (6 scalars)
- 总增加量: ~2K params

---

## Module 2: CCB (Causal Concept Bottleneck)

### 灵感来源
- EAP-IF (Li et al., ESWA 2026): 期望归因先验 + Disentangle-From-Normal
- XCDN (Vasquez et al., eJNDT 2026): 因果 vs 上下文因子解耦
- S2S Paradigm (ZJU, IEEE IES 2026): Signal-to-Semantics

### 6 个物理概念定义

| 概念 | 符号 | 物理意义 | 计算方式 |
|------|------|----------|----------|
| 冲击周期性 | c1 | 故障冲击脉冲的规律性程度 | 自相关函数峰值间隔的方差倒数 |
| 频带能量集中度 | c2 | 故障特征频带内的能量占比 | 特征频带能量 / 全频带能量 |
| 时域峭度 | c3 | 信号中冲击成分的强度 | 四阶矩 / 二阶矩的平方 |
| 包络调制深度 | c4 | 包络谱中特征频率幅值/背景噪声 | 特征频率峰值 / 邻域中位数 |
| 频谱熵 | c5 | 频谱的复杂度（噪声大则熵高） | 归一化频谱的香农熵 |
| 谐波衰减率 | c6 | 故障倍频的衰减速度（区分故障类型） | 倍频幅值序列的指数拟合衰减率 |

### 架构
1. **因果解耦层**：从融合特征中分离出
   - `z_causal`: 故障相关因子（输入概念瓶颈）
   - `z_context`: 工况/转速相关因子（输入域分类器）
   - 使用梯度反转层 (GRL) 驱动 z_causal 对域分类器的对抗训练

2. **概念瓶颈**：
   ```
   z_causal -> Linear(128->64) -> ReLU -> Linear(64->6) -> Sigmoid -> c_hat
   c_hat -> Linear(6->10) -> Softmax -> 故障分类
   ```
   分类器 **只能** 通过 6 个概念做决策，保证可解释性

3. **EAP 损失**：期望归因先验损失，惩罚模型在不合理的概念上产生高归因
   ```
   L_EAP = sum(max(0, A_k - E_k)^2)
   ```
   其中 A_k 是概念 k 的实际归因值，E_k 是期望归因值（由故障类型预设）

### 预设归因先验 (E_k)

| 故障类型 | c1 | c2 | c3 | c4 | c5 | c6 |
|----------|----|----|----|----|----|-----|
| NC (正常) | 0.0 | 0.1 | 0.1 | 0.0 | 0.8 | 0.0 |
| IF (内圈) | 0.8 | 0.7 | 0.6 | 0.7 | 0.2 | 0.5 |
| OF (外圈) | 0.9 | 0.8 | 0.7 | 0.8 | 0.2 | 0.6 |
| BF (滚动体) | 0.7 | 0.5 | 0.5 | 0.5 | 0.3 | 0.3 |

---

## Module 3: MVM-CDG (Multi-View Manifold Contrastive Domain Generalization)

### 灵感来源
- MDM-CA Net (Yang et al., Scientific Reports 2026): 时频域动态 K-NN 图 + 交叉注意力
- SAHGT (Han et al., Information Fusion 2026): 频谱感知异构图 Transformer
- 对比解耦 DG (Wang et al., MST 2026): HSIC + BYOL + 原型高斯三元组损失
- SAPCL (Zhang et al., ERX 2026): 频谱感知原型对比学习

### 组件

1. **动态 K-NN 图构建**
   - 在每个 batch 内，分别在时域特征和频域特征空间构建 K-NN 图 (K=5)
   - 图卷积 (GCN) 聚合邻域信息，过滤噪声引入的虚假连接
   - 两视角图特征通过交叉注意力融合

2. **HSIC 特征解耦**
   - 使用 Hilbert-Schmidt Independence Criterion 作为解耦损失
   - `L_HSIC = HSIC(z_class, z_domain)` -> 最小化类特征与域特征的依赖
   - 配合动态权重冗余约简 (来自 MST 2026 论文)

3. **概念空间监督对比损失**
   ```
   L_CC = -log( sum_{p in P(i)} exp(sim(c_hat_i, c_hat_p) / tau) /
                sum_{j != i} exp(sim(c_hat_i, c_hat_j) / tau) )
   ```
   在概念瓶颈输出 c_hat 上做对比学习，同类故障不同工况的概念激活一致化

4. **频谱感知原型学习**
   - 每类故障维护一个可学习的原型向量
   - L_proto = MSE(c_hat_i, proto_yi) + margin * max(0, delta - MSE(c_hat_i, proto_not_yi))
   - 原型通过动量更新

---

## Loss Function

### 总损失

```
L_total = lambda_cls * L_cls                # 分类损失 (标签平滑 CE)
        + lambda_recon * L_recon            # 信号重建损失 (保留)
        + lambda_physics * L_physics        # MCK 物理约束 (保留)
        + lambda_denoise * L_denoise        # 去噪模块损失 (新增)
        + lambda_EAP * L_EAP                # 期望归因先验损失 (新增)
        + lambda_concept * L_concept        # 概念瓶颈损失 (新增)
        + lambda_HSIC * L_HSIC              # HSIC 解耦损失 (新增)
        + lambda_CC * L_CC                  # 概念对比损失 (新增)
        + lambda_proto * L_proto            # 原型对比损失 (新增)
```

### 损失权重

| 损失 | 权重 | 说明 |
|------|------|------|
| L_cls | 8.0 | 主导，保留原权重 |
| L_recon | 0.2 | 保留 |
| L_physics | 0.1 | 保留 |
| L_denoise | 1.0 | 新增：MSE(clean, denoised) + SNR 估计误差 |
| L_EAP | 0.5 | 新增：归因先验一致性 |
| L_concept | 0.3 | 新增：概念真实性 (概念预测值 vs 物理计算值) |
| L_HSIC | 0.2 | 新增：特征解耦 |
| L_CC | 0.5 | 新增：概念空间对比 |
| L_proto | 0.3 | 新增：原型约束 |

---

## Training Strategy

### Phase 1: 预训练去噪模块 (5 epochs)
- 冻结 TF-PINN 主干，仅训练 SUD 模块
- 用含噪声的信号对 (clean, noisy) 做监督去噪
- L = MSE(denoised, clean) + SNR 估计正则

### Phase 2: 联合训练 (40 epochs)
- 解冻全部参数
- 随机 SNR in [-15, 5] dB 输入
- OneCycleLR 学习率调度
- 早停 patience=25

### Phase 3: 域泛化微调 (20 epochs)
- 多工况数据混合训练
- 激活 HSIC + CC + Proto 损失
- 域对抗训练 (GRL lambda 从 0->1 渐进)

---

## Datasets

### CWRU 数据集
- 4 工况: 0HP, 1HP, 2HP, 3HP
- 10 故障类别: NC, IF1-3, OF1-3, BF1-3
- 样本: SAMPLE_LENGTH=1024, 滑动步长=512

---

## Experiment Plan

### 实验 1: 强噪声鲁棒性 (核心)

| 维度 | 设置 |
|------|------|
| 数据集 | CWRU (0HP 训练, 0HP 测试) |
| SNR 范围 | clean, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10 dB |
| 噪声类型 | 高斯白噪声 |
| 重复次数 | 5 次 (mean±std) |
| 对比方法 | TF-PINN, WDCNN, ResNet-18, MDM-CA Net, LSR-Net, PFISTA-Net |

**输出**: Acc-SNR 曲线, 关键 SNR 混淆矩阵, 全量指标表

---

### 实验 2: 跨工况域泛化 (核心)

| 维度 | 设置 |
|------|------|
| 源域 | 每个工况轮流做源域 (4 tasks) |
| 目标域 | 其余 3 个工况 |
| SNR 条件 | clean, -4dB, -10dB |
| 重复次数 | 5 次 |

**输出**: 12 任务 DG 准确率热力图, t-SNE 特征/概念分布对比

---

### 实验 3: XAI 事前可解释性

**不做事后方法 (无 SHAP/LIME/Grad-CAM)**。只验证内建概念瓶颈的透明性：

| 指标 | 方法 |
|------|------|
| 概念归因一致性 | EAP 先验 vs 实际概念激活的 Spearman 相关 |
| 概念区分度 | 不同故障类型间概念激活 JS 散度 |
| 分类保真度 | 概念→分类 vs 特征→分类 准确率差 |

**可视化**:
- 概念激活雷达图 (各故障类型)
- 概念激活随 SNR 变化曲线
- t-SNE 概念空间分布

---

### 实验 4: 消融实验

| 变体 | 移除内容 |
|------|----------|
| Baseline (TF-PINN) | 全部新模块 |
| w/o SUD | Stockwell 去噪 |
| w/o CCB | 概念瓶颈 (直连分类器) |
| w/o EAP | 期望归因先验损失 |
| w/o MVM-CDG | DG 模块 |
| w/o HSIC | HSIC 解耦 |
| w/o Contrastive | 概念对比损失 |
| **Full DX-PINN** | — |

**输出**: 消融矩阵表, 贡献度柱状图

---

### 实验 5: 噪声类型泛化

| 噪声类型 | SNR 点 |
|----------|--------|
| 高斯白噪声 | -10 到 10 dB |
| 粉红噪声 (1/f) | -10 到 10 dB |
| 脉冲噪声 (1%, 5%) | -10 到 10 dB |
| 混合噪声 | -10 到 10 dB |

---

## 评估指标总结

| 实验 | 主要指标 | 目标 |
|------|----------|------|
| 噪声鲁棒性 | Acc@-10dB | > 90% |
| 跨工况泛化 | Avg Acc (12 tasks) @ -4dB | > 85% |
| XAI 质量 | 概念归因一致性 | > 0.7 |
| 消融实验 | 各模块贡献度 | 每模块 >= 1% |
| 噪声类型泛化 | Acc@-10dB (各类型) | > 85% |

---

## File Structure

```
├── config.py              # 扩展配置 (新增概念定义、DG 参数、去噪参数)
├── model/
│   ├── __init__.py
│   ├── sud.py             # Stockwell Unrolled Denoiser
│   ├── time_encoder.py    # 时域编码器 (保留)
│   ├── freq_encoder.py    # 频域编码器 (保留)
│   ├── fusion.py          # 交叉注意力融合 (保留)
│   ├── ccb.py             # 因果概念瓶颈
│   ├── mvm_cdg.py         # 多视角流形对比 DG
│   └── dx_pinn.py         # 主模型 (整合)
├── loss/
│   ├── __init__.py
│   ├── base_losses.py     # 分类/重建/物理损失 (保留)
│   ├── denoise_loss.py    # 去噪损失
│   ├── concept_loss.py    # 概念瓶颈 + EAP 损失
│   └── dg_loss.py         # HSIC + 对比 + 原型损失
├── data_loader.py          # 扩展多工况加载
├── train.py                # 三阶段训练
├── test.py                 # 评估 + XAI 可视化
├── xai/
│   ├── __init__.py
│   ├── concept_viz.py      # 概念激活可视化
│   └── attribution.py      # 归因分析
├── main.py
└── utils.py
```

---

## Implementation Order

1. `model/sud.py` — Stockwell 展开式去噪模块
2. `loss/denoise_loss.py` — 去噪损失函数
3. `model/ccb.py` — 因果概念瓶颈模块
4. `loss/concept_loss.py` — 概念损失 + EAP 损失
5. `model/mvm_cdg.py` — 多视角流形对比 DG 模块
6. `loss/dg_loss.py` — DG 相关损失
7. `model/dx_pinn.py` — 主模型整合
8. `config.py` — 扩展配置
9. `train.py` — 三阶段训练
10. `test.py` + `xai/` — 评估与可视化
11. `main.py` — 完整运行入口

---

## References

- EAP-IF: Li et al., "Expected Attribution Prior Guided Interpretable Framework", ESWA 2026
- PFISTA-Net: Zhong et al., "Physics-informed Adaptive ISTA Network", Neurocomputing 2026
- XCDN: Vasquez et al., "Contextual Fault Diagnosis via Causal Disentanglement and XAI", eJNDT 2026
- MDM-CA Net: Yang et al., "Multi-view Dynamic Manifold Reconstruction", Scientific Reports 2026
- LSR-Net: Lee et al., "Lightweight Strong Robustness Network", arXiv 2026
- SAHGT: Han et al., "Structure-Aware Heterogeneous Graph Transformer", Information Fusion 2026
- Contrastive Disentanglement DG: Wang et al., MST 2026
- SAPCL: Zhang et al., "Spectrum-Aware Prototypical Contrastive Learning", ERX 2026
- SAFATN: "Stockwell Weight Initialization + AFAT", EAAI 2025
- S2S: "Signal-to-Semantics LLM-Powered Fault Diagnosis", IEEE IES 2026
