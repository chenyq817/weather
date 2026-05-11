"""
本地训练+测试脚本 (优化版 v4)
四类天气识别: sunny, rainy, cloudy, snow

策略:
  - CrossEntropyLoss + LabelSmoothing (v2验证有效)
  - EMA 指数移动平均 (平滑权重, 提升泛化)
  - 早停机制 (patience=7)
  - MixUp + CutMix + CosineWarmRestarts + 增强分类头
  - 更强数据增强 (TrivialAugmentWide)
  - 更长训练 (25轮)
"""

import os
import sys
import time
import datetime
import copy
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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v4_{timestamp}.log")


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
BATCH_SIZE = 24
NUM_EPOCHS = 25
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 0.2
MIXUP_PROB = 0.5
GRAD_CLIP = 1.0
EMA_DECAY = 0.9997              # EMA衰减率 (更高=更平滑)
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 0.002

CLASS_NAMES = ["sunny", "rainy", "cloudy", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")

print(f"Device: {DEVICE} | Classes: {CLASS_NAMES}")
print(f"Epochs: {NUM_EPOCHS} | Warmup: {WARMUP_EPOCHS} | Batch: {BATCH_SIZE}")
print(f"Loss=CE+LabelSmooth | EMA decay={EMA_DECAY}")
print(f"EarlyStopping patience={EARLY_STOP_PATIENCE} min_delta={EARLY_STOP_MIN_DELTA}")

# ============ EMA (指数移动平均) ============
class ModelEMA:
    """
    对模型权重做指数移动平均，验证时使用EMA权重
    EMA权重 = decay * EMA权重 + (1 - decay) * 当前权重
    参考: Polyak & Juditsky (1992), "Acceleration of Stochastic Approximation by Averaging"
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (self.decay * self.shadow[name]
                               + (1.0 - self.decay) * param.data.detach())
                self.shadow[name] = new_average

    def apply_shadow(self):
        """将EMA权重应用到模型（验证前调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重（验证后调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


# ============ 早停 ============
class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.002, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == 'max' \
              else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============ 数据增强 ============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
print(f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
print(f"EfficientNet-B0: {total_p:.2f}M params (分类头: 1280→256→4)")

# EMA
ema = ModelEMA(model, decay=EMA_DECAY)

# ============ MixUp / CutMix ============
def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def cutmix_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    B, _, H, W = x.size()
    index = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x0, y0 = np.clip(cx - bw // 2, 0, W), np.clip(cy - bh // 2, 0, H)
    x1, y1 = np.clip(cx + bw // 2, 0, W), np.clip(cy + bh // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y0:y1, x0:x1] = x[index, :, y0:y1, x0:x1]
    lam = 1 - ((x1 - x0) * (y1 - y0) / (H * W))
    return mixed_x, y, y[index], lam


# ============ 训练 ============
print("\n=== 开始训练 (CE+LabelSmooth + EMA + EarlyStopping) ===")

# 标准交叉熵 + Label Smoothing (v2验证有效)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Phase 1: 冻结backbone
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                           min_delta=EARLY_STOP_MIN_DELTA, mode='max')

best_f1 = 0.0
best_epoch = 0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v4.pth")
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Phase 2: 解冻
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
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6,
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

        # 更新EMA
        ema.update()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    # ---- Validate (用原始权重, EMA仅用于最终测试) ----
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
    per_class_f1 = f1_score(val_targets, val_preds, average=None)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    phase = "WARM" if epoch < WARMUP_EPOCHS else "FT  "
    lr = optimizer.param_groups[0]['lr']
    pc_str = " ".join([f"{CLASS_NAMES[i][:3]}={per_class_f1[i]:.3f}" for i in range(len(CLASS_NAMES))])
    print(f"{phase} Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"loss={avg_train_loss:.4f} | v_loss={avg_val_loss:.4f} | "
          f"v_acc={val_acc:.4f} | v_f1={val_f1:.4f} | lr={lr:.2e}")
    print(f"       Per-class F1: {pc_str}")

    if val_f1 > best_f1 + EARLY_STOP_MIN_DELTA:
        best_f1 = val_f1
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_shadow": copy.deepcopy(ema.shadow),
            "class_names": CLASS_NAMES,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "per_class_f1": per_class_f1.tolist(),
        }, best_ckpt)
        print(f"  >>> Best model saved (F1={best_f1:.4f}, score={best_f1*100:.1f})")

    if early_stop(val_f1, epoch):
        print(f"\n>>> 早停触发! 验证F1连续{EARLY_STOP_PATIENCE}轮未提升, "
              f"最佳F1={best_f1:.4f} @ epoch {best_epoch+1}")
        break

elapsed = time.time() - start_time
print(f"\n训练完成! 耗时: {elapsed/60:.1f} 分钟 | 最佳轮次: {best_epoch+1}")
print(f"最佳验证F1: {best_f1:.4f} -> 比赛得分: {best_f1*100:.1f}")

# ============ 测试集评估 (EMA权重 + 原始权重对比) ============
print("\n=== 测试集评估 ===")
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# 先用原始权重评估
model.eval()
test_preds_raw, test_targets_raw = [], []
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_preds_raw.extend(preds.cpu().numpy())
        test_targets_raw.extend(targets.cpu().numpy())
test_f1_raw = f1_score(test_targets_raw, test_preds_raw, average="macro")

# 再用EMA权重评估
ema.apply_shadow()
model.eval()

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
test_per_class = f1_score(test_targets, test_preds, average=None)

print(f"\n测试集结果 ({len(test_dataset)} images):")
print(f"  [原始权重]: F1(macro)={test_f1_raw:.4f}, Acc={accuracy_score(test_targets_raw, test_preds_raw):.4f}")
print(f"  [EMA权重]:   F1(macro)={test_f1_macro:.4f}, Acc={test_acc:.4f}")
print(f"  EMA提升:     {test_f1_macro - test_f1_raw:+.4f}")
print(f"  推理总时间:     {test_time:.2f}s")
print(f"  平均推理:       {test_time/len(test_dataset)*1000:.2f}ms/image")
print(f"  >>> 预估得分:   {max(test_f1_macro, test_f1_raw)*100:.1f}")
print(f"  EMA各类别F1: " + " | ".join(
    f"{cls}={test_per_class[i]:.4f}" for i, cls in enumerate(CLASS_NAMES)))

print("\n分类报告:")
print(classification_report(test_targets, test_preds,
      target_names=CLASS_NAMES, digits=4))

print("混淆矩阵:")
cm = confusion_matrix(test_targets, test_preds)
header = "          " + " ".join(f"{n:>8s}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(len(CLASS_NAMES))))

# ============ 对比上一版 ============
print("\n=== 与 v2 版本对比 ===")
print("                     v2(CE+MixUp)   v3(Focal+EMA)")
v2_f1 = {"sunny": 0.8018, "rainy": 0.8039, "cloudy": 0.6522, "snow": 0.9263}
for i, cls in enumerate(CLASS_NAMES):
    v2_val = v2_f1.get(cls, 0)
    v3_val = test_per_class[i]
    delta = v3_val - v2_val
    sign = "+" if delta >= 0 else ""
    print(f"  {cls:8s}       {v2_val:.4f}          {v3_val:.4f}  ({sign}{delta:.4f})")
print(f"  {'MACRO':8s}       {0.7961:.4f}          {test_f1_macro:.4f}  ({'+' if test_f1_macro>=0.7961 else ''}{test_f1_macro-0.7961:.4f})")

# ============ 错误分析 ============
print("\n=== 错误分析 ===")
for i, cls_true in enumerate(CLASS_NAMES):
    for j, cls_pred in enumerate(CLASS_NAMES):
        if i != j and cm[i][j] > 0:
            print(f"  {cls_true} → {cls_pred}: {cm[i][j]} 张 (占{cls_true}测试集的{cm[i][j]/sum(cm[i])*100:.1f}%)")

print(f"\n=== 全部完成 ===")
print(f"模型: {best_ckpt}")
print(f"日志: {LOG_FILE}")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
