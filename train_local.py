"""
本地训练+测试脚本 (CPU, 超参数优化版)
四类天气识别: sunny, rainy, cloudy, snow
训练集翻倍(800/类) + MixUp + CutMix + CosineWarmRestarts
"""

import os
import sys
import time
import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_{timestamp}.log")


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

# ============ 超参数 (优化版) ============
IMAGE_SIZE = 224
BATCH_SIZE = 24
NUM_EPOCHS = 20
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3           # 增加mixup强度
CUTMIX_ALPHA = 0.2          # 新增cutmix
MIXUP_PROB = 0.6            # mixup/cutmix触发概率
GRAD_CLIP = 1.0
CLASS_NAMES = ["cloudy", "rain", "shine", "sunrise"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
print(f"Epochs: {NUM_EPOCHS} | Warmup: {WARMUP_EPOCHS} | Batch: {BATCH_SIZE}")
print(f"MixUp alpha={MIXUP_ALPHA} | CutMix alpha={CUTMIX_ALPHA}")
print(f"LabelSmoothing={LABEL_SMOOTHING} | WeightDecay={WEIGHT_DECAY}")

# ============ 数据增强 (强化版) ============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
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
print("\n=== 加载数据 ===")
train_dataset = WeatherDataset(TRAIN_DIR, transform=train_transform)
val_dataset = WeatherDataset(VAL_DIR, transform=val_transform)
test_dataset = WeatherDataset(TEST_DIR, transform=val_transform)

print(f"Train: {len(train_dataset)} images")
for cls in CLASS_NAMES:
    cnt = sum(1 for _, l in train_dataset.samples if CLASS_NAMES[l] == cls)
    print(f"  {cls}: {cnt}")
print(f"Val: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0)

# ============ 模型 ============
print("\n=== 构建模型 ===")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.35),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, NUM_CLASSES),
)
model = model.to(DEVICE)
total_p = sum(p.numel() for p in model.parameters()) / 1e6
print(f"EfficientNet-B0: {total_p:.2f}M params (增强分类头: 1280→256→4)")


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

# Phase 1: 冻结backbone
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

best_f1 = 0.0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model.pth")
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Phase 2: 解冻backbone
    if epoch == WARMUP_EPOCHS:
        print(f"\n>>> Epoch {epoch+1}: 解冻backbone，全模型微调")
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
        # CosineWarmRestarts: 周期性重启，更好跳出局部最优
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6,
        )

    # ---- Train ----
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # 混合增强策略
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

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

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
print("\n=== 测试集评估 ===")
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

print(f"\n测试集结果 ({len(test_dataset)} images):")
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

# ============ 每类推理速度 ============
print("\n推理速度 (每类10张):")
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    imgs = sorted(os.listdir(cls_dir))[:10]
    times = []
    for fname in imgs:
        image = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
        tensor = val_transform(image).unsqueeze(0).to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            _ = model(tensor)
        times.append((time.time() - t0) * 1000)
    print(f"  {cls_name}: avg={np.mean(times):.2f}ms")

print(f"\n=== 全部完成 ===")
print(f"模型: {best_ckpt}")
print(f"日志: {LOG_FILE}")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
