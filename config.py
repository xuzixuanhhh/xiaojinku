import torch
import os

# ===================== 数据集选择 =====================
DATASET = "CWRU"  # "CWRU" or "MFPT" or "XJTU"

# ===================== 核心路径配置 =====================
if DATASET == "CWRU":
    DATA_ROOT = "./CWRU Dataest"
elif DATASET == "MFPT":
    DATA_ROOT = "./MFPT"
else:
    DATA_ROOT = r"D:\BaiduNetdiskDownload\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets.part01\XJTU-SY_Bearing_Datasets"

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
if DATASET == "CWRU":
    FAULT_CLASSES = {
        "NC": 0, "IF1": 1, "IF2": 2, "IF3": 3,
        "OF1": 4, "OF2": 5, "OF3": 6,
        "BF1": 7, "BF2": 8, "BF3": 9,
    }
    BEARING_PARAMS = {
        "D": 39.04e-3, "d": 7.94e-3, "Z": 9, "alpha": 0, "fs": 12000,
    }
elif DATASET == "MFPT":
    FAULT_CLASSES = {"NC": 0, "IF1": 1, "IF2": 2, "OF1": 3, "OF2": 4}
    BEARING_PARAMS = {
        "D": 31.623e-3, "d": 5.969e-3, "Z": 8, "alpha": 0, "fs": 48828,
    }
else:  # XJTU-SY: LDK UER204, 4 classes (NC/IF/OF/CF), cond2=37.5Hz11kN
    FAULT_CLASSES = {"NC": 0, "IF": 1, "OF": 2, "CF": 3}
    BEARING_PARAMS = {
        "D": 34.55e-3, "d": 7.92e-3, "Z": 8, "alpha": 0, "fs": 25600,
    }
NUM_CLASSES = len(FAULT_CLASSES)

if DATASET == "CWRU":
    SHAFT_SPEED = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
elif DATASET == "MFPT":
    SHAFT_SPEED = {0: 1500, 1: 1500, 2: 1500, 3: 1500}
else:  # XJTU-SY: 3 conditions, key 3 mapped for model compat
    SHAFT_SPEED = {0: 2100, 1: 2250, 2: 2400, 3: 2250}  # RPM

TRAIN_SNR_RANGE = (-10, 10)

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