"""
天气分类竞赛配置文件
智海算法调优 - 四类天气识别 (晴天/雨天/阴天/雪天)
"""

import os
import torch

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")          # 数据集目录
TRAIN_DIR = os.path.join(DATA_DIR, "train")         # 训练集
VAL_DIR = os.path.join(DATA_DIR, "val")             # 验证集
TEST_DIR = os.path.join(DATA_DIR, "test")           # 测试集
OUTPUT_DIR = os.path.join(BASE_DIR, "output")       # 输出目录
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # 模型保存
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")          # 日志

# 自动创建目录
for d in [OUTPUT_DIR, CKPT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 天气类别 ============
# 根据比赛数据调整，常见四类:
CLASS_NAMES = ["sunny", "rainy", "cloudy", "snow"]
NUM_CLASSES = len(CLASS_NAMES)

# ============ 模型配置 ============
MODEL_NAME = "efficientnet_b0"  # 可选: mobilenet_v3_large, resnet50, densenet121
PRETRAINED = True
DROPOUT_RATE = 0.3              # 分类头dropout

# ============ 训练配置 ============
BATCH_SIZE = 24                 # CPU本地训练，不宜太大
NUM_EPOCHS = 15                 # 本地CPU训练测试
WARMUP_EPOCHS = 2               # 预热轮数（仅训练分类头）
IMAGE_SIZE = 224                # 输入图像尺寸

# 优化器
LEARNING_RATE = 1e-3            # 分类头学习率
FINE_TUNE_LR = 1e-4             # 微调阶段学习率 (backbone)
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adamw"             # adamw / sgd

# 学习率调度
LR_SCHEDULER = "cosine"         # cosine / step / plateau
LR_MIN = 1e-6
WARMUP_EPOCHS_SCHEDULER = 3     # 学习率warmup

# 正则化
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2               # mixup强度，0表示不使用
CUTMIX_ALPHA = 0.0              # cutmix强度，0表示不使用
GRAD_CLIP = 1.0                 # 梯度裁剪

# 早停
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "f1"    # f1 / loss

# ============ 数据增强 ============
AUGMENTATION = {
    "random_crop_scale": (0.8, 1.0),
    "random_rotation": 20,          # 随机旋转角度
    "random_h_flip": 0.5,           # 水平翻转概率
    "random_v_flip": 0.1,           # 垂直翻转概率
    "color_jitter": {               # 颜色抖动
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.1,
    },
    "random_erasing_p": 0.2,        # 随机擦除
    "auto_augment": True,           # 自动增强
}

# 测试时增强 (TTA)
TTA_TIMES = 5  # TTA次数，0表示不使用

# ============ 硬件配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0                    # CPU训练用0（避免多进程开销）
USE_AMP = False                    # CPU无混合精度训练
AMP_DTYPE = "float16"              # bfloat16 / float16

# ============ 推理优化 ============
USE_TORCHSCRIPT = True             # TorchScript编译加速推理
USE_QUANTIZATION = False           # 量化（视平台支持情况开启）
INFERENCE_BATCH_SIZE = 128         # 推理batch size
