"""
优化v2训练: 定向CLAHE + 边缘密度通道 + Focal Loss
根因分析后的针对性修复:
  1. CLAHE只对阴天应用 (避免损害晴天特征)
  2. 边缘密度作为第4通道输入 (阴天低边缘 vs 晴天高边缘)
  3. Focal Loss 聚焦困难样本 (阴天是最难的类别)
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

from model_v2 import build_edge_model, FocalLoss, count_parameters

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v2_{timestamp}.log")


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

# ============ 超参数 ============
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 5e-5
WEIGHT_DECAY = 3e-4
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 0.3
MIXUP_PROB = 0.8
GRAD_CLIP = 1.0
DROPOUT = 0.5
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
CLOUDY_IDX = CLASS_NAMES.index("cloudy")
DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
print(f"Epochs: {NUM_EPOCHS} | Warmup: {WARMUP_EPOCHS} | Batch: {BATCH_SIZE}")
print(f"Focal Loss gamma={FOCAL_GAMMA} | MixUp alpha={MIXUP_ALPHA}")
print(f"优化v2: 定向CLAHE(仅阴天) + 边缘通道 + FocalLoss")

# ============ 定向CLAHE (仅阴天) ============
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
CLAHE_CLOUDY_PROB = 0.8  # 阴天80%概率应用CLAHE


def apply_clahe_to_image(img):
    """对PIL Image应用CLAHE (LAB/L通道)"""
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_np)


def extract_edge_map(img_pil):
    """提取边缘密度图 (Sobel梯度幅值)"""
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x**2 + sobel_y**2)
    # 归一化到 [0, 255]
    edge = np.clip(edge / edge.max() * 255, 0, 255).astype(np.uint8) if edge.max() > 0 else edge
    return Image.fromarray(edge)


# ============ 数据增强管线 ============
train_rgb_transform = transforms.Compose([
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

val_rgb_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 边缘图独立管线 (不应用颜色增强)
edge_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    # 边缘图单通道, 用均值归一化
    transforms.Normalize(mean=[0.5], std=[0.25]),
])


class EdgeWeatherDataset(Dataset):
    """4通道数据集: RGB(3ch) + 边缘密度(1ch)
    定向CLAHE: 仅对阴天(cloudy)应用
    """
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((
                            os.path.join(class_dir, fname),
                            self.class_to_idx[class_name],
                            class_name,
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        # 定向CLAHE: 仅训练时对阴天应用
        if self.is_train and class_name == "cloudy" and random.random() < CLAHE_CLOUDY_PROB:
            image = apply_clahe_to_image(image)

        # 提取边缘图 (在RGB增强之前, 保持边缘信息不被颜色抖动破坏)
        edge = extract_edge_map(image)

        # RGB增强
        if self.is_train:
            rgb = train_rgb_transform(image)
        else:
            rgb = val_rgb_transform(image)

        # 边缘图增强 (同步空间变换)
        edge = edge_transform(edge)

        # 拼接为4通道
        combined = torch.cat([rgb, edge], dim=0)  # (4, H, W)
        return combined, label


# ============ 加载数据 ============
print("\n=== 加载数据 (weather_clean + 边缘通道) ===")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

full_train = EdgeWeatherDataset(TRAIN_DIR, is_train=True)
test_dataset = EdgeWeatherDataset(TEST_DIR, is_train=False)

class_counts = {}
for _, label, cls_name in full_train.samples:
    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
print(f"训练集: {len(full_train)} 张")
for cls in CLASS_NAMES:
    print(f"  {cls}: {class_counts.get(cls, 0)} 张")
print(f"测试集: {len(test_dataset)} 张")

# 验证集切分
from torch.utils.data import random_split
val_size = max(1, int(0.15 * len(full_train)))
train_size = len(full_train) - val_size
train_dataset, _ = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
full_for_val = EdgeWeatherDataset(TRAIN_DIR, is_train=False)
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
print("\n=== 构建模型 (4通道输入) ===")
model = build_edge_model(num_classes=NUM_CLASSES, dropout=DROPOUT)
model = model.to(DEVICE)
total_p, trainable_p = count_parameters(model)
print(f"EfficientNet-B0(4ch): {total_p/1e6:.2f}M params ({trainable_p/1e6:.2f}M trainable)")
print(f"  输入: RGB(3ch) + Edge(1ch)")
print(f"  分类头: 1280→256→{NUM_CLASSES}")


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
print("\n=== 开始训练 (Focal Loss) ===")
criterion = FocalLoss(gamma=FOCAL_GAMMA)

# Phase 1: 冻结backbone，只训分类头
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_f1 = 0.0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v2.pth")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Phase 2: 解冻全部
    if epoch == WARMUP_EPOCHS:
        print(f"\n>>> Epoch {epoch+1}: 解冻全部参数")
        for param in model.parameters():
            param.requires_grad = True
        head_params = [p for n, p in model.named_parameters()
                       if "classifier" in n and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters()
                           if "classifier" not in n and p.requires_grad]
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
print(f"最佳验证F1: {best_f1:.4f}")

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

print("\n分类报告:")
print(classification_report(test_targets, test_preds,
      target_names=CLASS_NAMES, digits=4))

print("混淆矩阵:")
cm = confusion_matrix(test_targets, test_preds)
header = "          " + " ".join(f"{n:>8s}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(len(CLASS_NAMES))))

# ============ 与前两版对比 ============
print("\n" + "=" * 60)
print("=== 三个版本对比 ===")
print("=" * 60)
baseline = {"cloudy": 0.6801, "rain": 0.8550, "sunny": 0.7740, "snow": 0.9147, "macro": 0.8060}
v1 = {"cloudy": 0.6859, "rain": 0.8593, "sunny": 0.7619, "snow": 0.9171, "macro": 0.8060}
per_class = f1_score(test_targets, test_preds, average=None)
v2 = {
    "cloudy": per_class[CLASS_NAMES.index("cloudy")],
    "rain": per_class[CLASS_NAMES.index("rain")],
    "sunny": per_class[CLASS_NAMES.index("sunny")],
    "snow": per_class[CLASS_NAMES.index("snow")],
    "macro": test_f1_macro,
}

print(f"{'类别':<12} {'Baseline':<12} {'v1(CLAHE+CBAM)':<16} {'v2(边缘+Focal)':<16}")
print("-" * 56)
for cls in CLASS_NAMES:
    print(f"  {cls:<10} {baseline[cls]:.4f}       {v1[cls]:.4f}           {v2[cls]:.4f}")
print(f"  {'macro avg':<10} {baseline['macro']:.4f}       {v1['macro']:.4f}           {v2['macro']:.4f}")

# cloudy混淆分析
cm_baseline = [[135, 17, 44, 4], [25, 171, 2, 2], [32, 3, 161, 4], [5, 9, 9, 177]]
print(f"\n阴天↔晴天混淆对比:")
print(f"  Baseline: cloudy→sunny={cm_baseline[0][2]}, sunny→cloudy={cm_baseline[2][0]}")
print(f"  v2:       cloudy→sunny={cm[0][2]}, sunny→cloudy={cm[2][0]}")

print(f"\n=== 全部完成 ===")
print(f"模型: {best_ckpt}")
print(f"日志: {LOG_FILE}")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
