import torch
import os

# ===================== 核心路径配置（仅需修改此处！）=====================
# 数据集根目录：要求根目录下有 0HP、1HP、2HP、3HP 四个文件夹
# 每个工况文件夹内有10个故障子文件夹：NC、IF1、IF2、IF3、OF1、OF2、OF3、RF1、RF2、RF3
# 每个子文件夹内有1个对应的.mat文件
DATA_ROOT = "./CWRU Dataest"  # 替换为你的数据集根目录

# 模型与结果保存路径
SAVE_PATH = "./results"
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "tf_pinn_model.pth")
LOG_SAVE_PATH = os.path.join(SAVE_PATH, "train_log.csv")
os.makedirs(SAVE_PATH, exist_ok=True)

# ===================== 数据集核心参数 =====================
# 工况选择：0=0HP,1=1HP,2=2HP,3=3HP，训练默认用0HP，变负载测试用其他工况
TRAIN_WORK_CONDITION = 0
TEST_WORK_CONDITIONS = [0,1,2,3]

# 故障类别与标签映射
FAULT_CLASSES = {
    "NC": 0,   # 正常
    "IF1": 1,  # 内圈故障 0.1778mm
    "IF2": 2,  # 内圈故障 0.3556mm
    "IF3": 3,  # 内圈故障 0.5334mm
    "OF1": 4,  # 外圈故障 0.1778mm
    "OF2": 5,  # 外圈故障 0.3556mm
    "OF3": 6,  # 外圈故障 0.5334mm
    "BF1": 7,  # 滚动体故障 0.1778mm
    "BF2": 8,  # 滚动体故障 0.3556mm
    "BF3": 9   # 滚动体故障 0.5334mm
}
NUM_CLASSES = len(FAULT_CLASSES)

# 轴承核心几何参数（严格匹配CWRU 6205-2RS轴承，论文表1）
BEARING_PARAMS = {
    "D": 39.04e-3,    # 节圆直径，单位m
    "d": 7.94e-3,     # 滚动体直径，单位m
    "Z": 9,            # 滚动体个数
    "alpha": 0,        # 接触角，单位rad
    "fs": 12000,       # 采样频率，单位Hz
}

# 各工况对应的轴转速（CWRU官方标准值）
SHAFT_SPEED = {
    0: 1797,  # 0HP 转速 rpm
    1: 1772,  # 1HP 转速 rpm
    2: 1750,  # 2HP 转速 rpm
    3: 1730   # 3HP 转速 rpm
}

# 样本参数
SAMPLE_LENGTH = 1024
SLIDING_STEP = 1024   # Non-overlapping (MS-TCANet standard)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1       # 验证集占比
TEST_RATIO = 0.2      # 测试集占比

# ===================== 模型训练参数 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # GPU can handle larger batches
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 30

# 网络结构参数（严格匹配论文表3）
ENCODER_LAYERS = 6
ENCODER_HIDDEN_DIM = 128
DROPOUT_RATE = 0.2
CLASSIFIER_HIDDEN_DIM = 64

# ===================== 损失函数权重 =====================
LOSS_WEIGHTS = {
    "lambda_cls": 8.0,        # 分类损失主导，优先学习故障判别边界
    "lambda_recon": 0.2,      # 归一化重建辅助，避免过拟合干净幅值细节
    "lambda_fault_freq": 1.5, # 强化故障特征频率约束
    "lambda_physics": 0.1,    # 归一化物理残差仅作弱约束
    "lambda_mck_reg": 0.02,   # MCK 参数正则仅防止漂移
}

# ===================== 强噪声实验参数 =====================
SNR_LEVELS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
TRAIN_SNR_RANGE = (-15, 5)
NOISE_BANDWIDTH = 50
HARMONIC_NUM = 4

# ===================== DX-PINN 新增参数 =====================
NUM_CONCEPTS = 6
EAP_LAMBDA = 0.5
CONCEPT_LAMBDA = 0.3
DENOISE_LAMBDA = 1.0
NUM_ISTA_LAYERS = 3
SINC_KERNEL_SIZE = 15
SINC_NUM_KERNELS = 64
DG_LAMBDA = 1.0
HSIC_LAMBDA = 0.2
CC_LAMBDA = 0.5
PROTO_LAMBDA = 0.3
TEMPERATURE = 0.1
K_NN = 5
GRL_WARMUP_EPOCHS = 10
DG_TRAIN_CONDITIONS = [0, 1, 2, 3]
NOISE_TYPES = ["gaussian", "pink", "impulse", "mixed"]