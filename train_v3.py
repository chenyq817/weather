"""
v5 GPU训练: v3配置 + GPU加速
策略: EfficientNet-B2 + 260px + 40epoch + EarlyStopping + AMP
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

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v5_gpu_{timestamp}.log")


class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

sys.stdout = Logger(LOG_FILE)
print(f"日志: {LOG_FILE}")

# ============ 超参数 ============
IMAGE_SIZE = 260
BATCH_SIZE = 64            # GPU 可以跑大 batch
NUM_EPOCHS = 40
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 5e-5
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 0.2
MIXUP_PROB = 0.7
GRAD_CLIP = 1.0
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")
USE_AMP = True

EARLY_STOP_PATIENCE = 10

print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)}) | Size: {IMAGE_SIZE}px | Batch: {BATCH_SIZE}")
print(f"Model: EfficientNet-B2 | Epochs: {NUM_EPOCHS} | EarlyStop: {EARLY_STOP_PATIENCE} | AMP: {USE_AMP}")
print(f"MixUp={MIXUP_ALPHA} CutMix={CUTMIX_ALPHA} LabelSmooth={LABEL_SMOOTHING}")


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

# ============ 数据增强 ============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1),
    transforms.RandomEqualize(p=0.3),          # ★ 随机直方图均衡(比CLAHE更安全)
    transforms.RandomGrayscale(p=0.05),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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
        for cls in CLASS_NAMES:
            d = os.path.join(root_dir, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d, f), self.class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        if self.transform: img = self.transform(img)
        return img, l


# ============ 数据加载 ============
print("\n=== 加载数据 ===")
full_train = WeatherDataset(os.path.join(DATA_DIR, "train"), train_transform)
test_dataset = WeatherDataset(os.path.join(DATA_DIR, "test"), val_transform)

for cls in CLASS_NAMES:
    cnt = sum(1 for _, l in full_train.samples if CLASS_NAMES[l]==cls)
    print(f"  train/{cls}: {cnt}")
print(f"  test: {len(test_dataset)}")

from torch.utils.data import random_split
vs = max(1, int(0.15 * len(full_train)))
ts = len(full_train) - vs
train_dataset, _ = random_split(full_train, [ts, vs], generator=torch.Generator().manual_seed(42))
full_v = WeatherDataset(os.path.join(DATA_DIR, "train"), val_transform)
_, val_dataset = random_split(full_v, [ts, vs], generator=torch.Generator().manual_seed(42))
print(f"train={ts} val={vs} test={len(test_dataset)}")

tl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
tl_test = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============ 模型: EfficientNet-B2 ============
print("\n=== EfficientNet-B2 ===")
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
inf = model.classifier[1].in_features  # 1408
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(inf, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(256, NUM_CLASSES),
)
model = model.to(DEVICE)
tp = sum(p.numel() for p in model.parameters())/1e6
print(f"  Params: {tp:.2f}M | Head: {inf}→256→{NUM_CLASSES}")

# ============ MixUp/CutMix ============
def mixup(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    B, _, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W*np.sqrt(1-lam)), int(H*np.sqrt(1-lam))
    x0, y0 = np.clip(cx-bw//2,0,W), np.clip(cy-bh//2,0,H)
    x1, y1 = np.clip(cx+bw//2,0,W), np.clip(cy+bh//2,0,H)
    mx = x.clone(); mx[:,:,y0:y1,x0:x1] = x[idx,:,y0:y1,x0:x1]
    return mx, y, y[idx], 1 - (x1-x0)*(y1-y0)/(H*W)

# ============ 训练 ============
print("\n=== 训练 ===")
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Phase 1: 冻结backbone
for n, p in model.named_parameters():
    p.requires_grad = "classifier" in n

opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)

best_f1, best_ep = 0.0, 0
scaler = torch.amp.GradScaler() if USE_AMP else None
early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v5_gpu.pth")
t0 = time.time()

for ep in range(NUM_EPOCHS):
    if ep == WARMUP_EPOCHS:
        print(f"\n>>> Epoch {ep+1}: 解冻backbone")
        for p in model.parameters(): p.requires_grad = True
        hp = [p for n,p in model.named_parameters() if "classifier" in n]
        bp = [p for n,p in model.named_parameters() if "classifier" not in n]
        opt = optim.AdamW([{"params":hp,"lr":LEARNING_RATE},
                           {"params":bp,"lr":FINE_TUNE_LR}], weight_decay=WEIGHT_DECAY)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

    model.train(); tl_ = 0.0
    for x, y in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_PROB
        if ma:
            if torch.rand(1).item() < 0.6:
                x, ya, yb, lam = mixup(x, y, MIXUP_ALPHA)
            else:
                x, ya, yb, lam = cutmix(x, y, CUTMIX_ALPHA)

        opt.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x)
            loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        tl_ += loss.item()

    sch.step()

    # Validate
    model.eval(); vl_, preds, tgts = 0.0, [], []
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            vl_ += criterion(out, y).item()
            preds.extend(out.argmax(1).cpu().numpy())
            tgts.extend(y.cpu().numpy())

    acc = accuracy_score(tgts, preds)
    f1 = f1_score(tgts, preds, average="macro")
    lr = opt.param_groups[0]['lr']
    ph = "WARM" if ep < WARMUP_EPOCHS else "FT  "
    print(f"{ph} E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(tl):.4f} | "
          f"vloss={vl_/len(vl):.4f} | acc={acc:.4f} | f1={f1:.4f} | lr={lr:.2e}")

    if f1 > best_f1:
        best_f1, best_ep = f1, ep
        torch.save({"epoch":ep,"model":model.state_dict(),"opt":opt.state_dict(),
                    "classes":CLASS_NAMES,"size":IMAGE_SIZE,"f1":f1,"acc":acc}, best_ckpt)
        print(f"  >>> Best (F1={best_f1:.4f})")

    if early_stop(f1):
        print(f"\n>>> 早停触发! {EARLY_STOP_PATIENCE}轮未提升, 停止于 Epoch {ep+1}, 最佳 F1={best_f1:.4f} @ Epoch {best_ep+1}")
        break

print(f"\n训练完成: {time.time()-t0:.0f}s | Best val F1={best_f1:.4f} @ Epoch {best_ep+1}")

# ============ 测试 ============
print("\n=== 测试 ===")
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

preds, tgts = [], []
t0 = time.time()
with torch.no_grad():
    for x, y in tl_test:
        out = model(x.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
tt = time.time()-t0

acc = accuracy_score(tgts, preds)
f1m = f1_score(tgts, preds, average="macro")
per_f1 = f1_score(tgts, preds, average=None)

print(f"Accuracy: {acc:.4f} | F1 macro: {f1m:.4f} | Time: {tt:.1f}s")
print(f"\n分类报告:")
print(classification_report(tgts, preds, target_names=CLASS_NAMES, digits=4))

print("混淆矩阵:")
cm = confusion_matrix(tgts, preds)
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(NUM_CLASSES)))

# ============ 对比 ============
print("\n=== 全部版本对比 ===")
base = {"cloudy":0.6801,"rain":0.8550,"sunny":0.7740,"snow":0.9147,"macro":0.8060}
v1 =  {"cloudy":0.6859,"rain":0.8593,"sunny":0.7619,"snow":0.9171,"macro":0.8060}
print(f"{'类别':<12} {'Baseline':<10} {'v1CLAHE+CBAM':<14} {'v3B2+EQ+260':<14}")
print("-"*50)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<10} {base[cls]:.4f}      {v1[cls]:.4f}         {per_f1[i]:.4f}")
print(f"  {'macro':<10} {base['macro']:.4f}      {v1['macro']:.4f}         {f1m:.4f}")

print(f"\n阴天↔晴天混淆 (关键):")
print(f"  Baseline: cloudy→sunny=44, sunny→cloudy=32")
print(f"  v3:       cloudy→sunny={cm[0][2]}, sunny→cloudy={cm[2][0]}")

print(f"\n模型: {best_ckpt}")
print(f"日志: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
