"""
优化训练脚本: CLAHE预处理 + CBAM注意力 + HSV饱和度分支
针对阴天(cloudy)分类优化

优化点:
  1. CLAHE自适应直方图均衡化: 增强阴天图片局部对比度
  2. CBAM注意力: 通道+空间双注意力, 聚焦判别区域
  3. HSV分支: 利用阴天S(饱和度)<30的关键特征区间
"""

import os
import sys
import time
import datetime
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from model_optimized import OptimizedWeatherModel, count_parameters

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_optimized_{timestamp}.log")


class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)
print(f"日志: {LOG_FILE}")

# ============ 超参数 (与baseline一致, 公平对比) ============
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 5e-5
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.15
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 0.3
MIXUP_PROB = 0.8
GRAD_CLIP = 1.0
DROPOUT = 0.5
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
print(f"Epochs: {NUM_EPOCHS} | Warmup: {WARMUP_EPOCHS} | Batch: {BATCH_SIZE}")
print(f"MixUp alpha={MIXUP_ALPHA} | CutMix alpha={CUTMIX_ALPHA} | MixProb={MIXUP_PROB}")
print(f"LabelSmoothing={LABEL_SMOOTHING} | WeightDecay={WEIGHT_DECAY}")
print(f"优化: CLAHE + CBAM + HSV")

# ============ CLAHE 预处理 ============
class CLAHEEnhance:
    """CLAHE自适应直方图均衡化 — LAB色彩空间L通道增强
    概率性应用(prob=0.6), 让模型见过增强和未增强两种图片
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), prob=0.6):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img_np = np.array(img)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_np)
        return img


# ============ 数据增强管线 ============
train_transform = transforms.Compose([
    CLAHEEnhance(clip_limit=2.0, tile_grid_size=(8, 8), prob=0.6),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.1),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.35),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((
                            os.path.join(class_dir, fname),
                            self.class_to_idx[class_name]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


# ============ 加载数据 ============
print("\n=== 加载数据 (weather_clean) ===")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

full_train = WeatherDataset(TRAIN_DIR, transform=train_transform)
test_dataset = WeatherDataset(TEST_DIR, transform=val_transform)

class_counts = {}
for _, label in full_train.samples:
    cls = CLASS_NAMES[label]
    class_counts[cls] = class_counts.get(cls, 0) + 1
print(f"训练集: {len(full_train)} 张")
for cls in CLASS_NAMES:
    print(f"  {cls}: {class_counts.get(cls, 0)} 张")
print(f"测试集: {len(test_dataset)} 张")

# 切分验证集
from torch.utils.data import random_split
val_size = max(1, int(0.15 * len(full_train)))
train_size = len(full_train) - val_size
train_dataset, _ = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
full_for_val = WeatherDataset(TRAIN_DIR, transform=val_transform)
_, val_dataset = random_split(
    full_for_val, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
print(f"切分: train={train_size}, val={val_size}, test={len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============ 模型 ============
print("\n=== 构建优化模型 ===")
model = OptimizedWeatherModel(num_classes=NUM_CLASSES, dropout=DROPOUT, hsv_dim=64)
model = model.to(DEVICE)
total_p, trainable_p = count_parameters(model)
print(f"OptimizedModel: {total_p/1e6:.2f}M params ({trainable_p/1e6:.2f}M trainable)")
print(f"  Backbone: EfficientNet-B0 + CBAM + HSV({64}d)")
print(f"  分类头: (1280+64)→256→{NUM_CLASSES}")


# ============ MixUp / CutMix ============
def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bw = int(W * np.sqrt(1 - lam))
    bh = int(H * np.sqrt(1 - lam))
    x0 = np.clip(cx - bw // 2, 0, W)
    y0 = np.clip(cy - bh // 2, 0, H)
    x1 = np.clip(cx + bw // 2, 0, W)
    y1 = np.clip(cy + bh // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y0:y1, x0:x1] = x[index, :, y0:y1, x0:x1]
    lam = 1 - ((x1 - x0) * (y1 - y0) / (H * W))
    return mixed_x, y, y[index], lam


# ============ 训练 ============
print("\n=== 开始训练 ===")
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Phase 1: 冻结backbone, 只训分类头和HSV分支
for name, param in model.named_parameters():
    if "features" in name or "cbam" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_f1 = 0.0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_optimized.pth")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Phase 2: 解冻全部
    if epoch == WARMUP_EPOCHS:
        print(f"\n>>> Epoch {epoch+1}: 解冻全部参数，全模型微调")
        for param in model.parameters():
            param.requires_grad = True
        head_params = []
        backbone_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "features" in n or "cbam" in n:
                backbone_params.append(p)
            else:
                head_params.append(p)
        optimizer = optim.AdamW([
            {"params": head_params, "lr": LEARNING_RATE},
            {"params": backbone_params, "lr": FINE_TUNE_LR},
        ], weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=2, eta_min=1e-6,
        )

    # ---- Train ----
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        if epoch >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_PROB:
            if torch.rand(1).item() < 0.6:
                images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            mixup_active = True
        else:
            mixup_active = False

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        if mixup_active:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro")

    phase = "WARM" if epoch < WARMUP_EPOCHS else "FT  "
    lr = optimizer.param_groups[0]['lr']
    print(f"{phase} Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"loss={avg_train_loss:.4f} | "
          f"val_loss={avg_val_loss:.4f} | "
          f"val_acc={val_acc:.4f} | "
          f"val_f1={val_f1:.4f} | "
          f"lr={lr:.2e}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "val_f1": val_f1,
            "val_acc": val_acc,
        }, best_ckpt)
        print(f"  >>> Best model saved (F1={best_f1:.4f}, score={best_f1*100:.1f})")

elapsed = time.time() - start_time
print(f"\n训练完成! 耗时: {elapsed/60:.1f} 分钟")
print(f"最佳验证F1: {best_f1:.4f} -> 比赛得分: {best_f1*100:.1f}")

# ============ 测试集评估 ============
print("\n" + "=" * 60)
print("=== 测试集评估 ===")
print("=" * 60)
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

test_preds, test_targets = [], []
test_start = time.time()

with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

test_time = time.time() - test_start
test_acc = accuracy_score(test_targets, test_preds)
test_f1_macro = f1_score(test_targets, test_preds, average="macro")
test_f1_weighted = f1_score(test_targets, test_preds, average="weighted")

print(f"\n测试集结果 ({len(test_dataset)} 张):")
print(f"  Accuracy:       {test_acc:.4f}")
print(f"  F1 (macro):     {test_f1_macro:.4f}")
print(f"  F1 (weighted):  {test_f1_weighted:.4f}")
print(f"  推理总时间:     {test_time:.2f}s")
print(f"  平均推理:       {test_time/len(test_dataset)*1000:.2f}ms/image")
print(f"  >>> 预估得分:   {test_f1_macro*100:.1f}")

print("\n分类报告:")
print(classification_report(test_targets, test_preds,
      target_names=CLASS_NAMES, digits=4))

print("混淆矩阵:")
cm = confusion_matrix(test_targets, test_preds)
header = "          " + " ".join(f"{n:>8s}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(len(CLASS_NAMES))))

print("\n各类别F1:")
per_class_f1 = f1_score(test_targets, test_preds, average=None)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: {per_class_f1[i]:.4f}")

# ============ 与 Baseline 对比 ============
print("\n" + "=" * 60)
print("=== 优化前后对比 ===")
print("=" * 60)
baseline = {"cloudy": 0.6801, "rain": 0.8550, "sunny": 0.7740, "snow": 0.9147, "macro": 0.8060}
print(f"{'类别':<12} {'Baseline':<12} {'优化后':<12} {'变化':<12}")
print("-" * 48)
for i, cls in enumerate(CLASS_NAMES):
    diff = per_class_f1[i] - baseline[cls]
    sign = "+" if diff > 0 else ""
    print(f"  {cls:<10} {baseline[cls]:.4f}       {per_class_f1[i]:.4f}       {sign}{diff:.4f}")
diff_macro = test_f1_macro - baseline["macro"]
sign = "+" if diff_macro > 0 else ""
print(f"  {'macro avg':<10} {baseline['macro']:.4f}       {test_f1_macro:.4f}       {sign}{diff_macro:.4f}")

print(f"\n=== 全部完成 ===")
print(f"模型: {best_ckpt}")
print(f"日志: {LOG_FILE}")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
