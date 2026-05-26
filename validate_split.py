"""
方案A: 新seed重划分验证 — 验证v8提升是否真实
1. 合并全部清洗数据, 用seed=123重新划分 train(800/类) / test(200/类)
2. 用v8配方训练1个模型
3. 对比seed=42的test结果 vs seed=123的test结果
4. 如果F1仍在84%+附近 → v8提升真实, 非过拟合
"""
import os, sys, time, datetime, random, shutil
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"validate_split123_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout; self.log = open(p, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

# ============ Config (same as v8) ============
IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS = 260, 64, 40
WARMUP_EPOCHS = 3
LR, FINE_LR, WD = 1e-3, 5e-5, 3e-4
LABEL_SMOOTH, MIXUP_A, CUTMIX_A, MIXUP_P = 0.1, 0.3, 0.2, 0.7
GRAD_CLIP = 1.0; EARLY_PATIENCE = 10
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda"); USE_AMP = True
NEW_SEED = 123  # different from original seed=42
VAL_RATIO = 0.15

print(f"Device: {DEVICE} | {IMAGE_SIZE}px | Batch={BATCH_SIZE}")
print(f"New split seed: {NEW_SEED} (original was 42)")
print(f"Model: EfficientNet-B2, same v8 augmentations")

# ============ Augmentation (v8 tuned) ============
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomEqualize(p=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============ TTA ============
TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def tta_predict(model, img):
    img_resized = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_resized)
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            tensor = norm(to_tensor(c)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs.append(torch.softmax(model(tensor), 1))
    return torch.cat(probs).mean(0, keepdim=True)

# ============ Dataset ============
class WeatherDS(Dataset):
    def __init__(self, root, transform=None):
        self.root = root; self.tf = transform
        self.samples = []; self.c2i = {n: i for i, n in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d, f), self.c2i[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        if self.tf: img = self.tf(img)
        return img, l

# ============ MixUp / CutMix ============
def mixup(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    B, _, H, W = x.size(); idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x0, y0 = np.clip(cx - bw // 2, 0, W), np.clip(cy - bh // 2, 0, H)
    x1, y1 = np.clip(cx + bw // 2, 0, W), np.clip(cy + bh // 2, 0, H)
    mx = x.clone(); mx[:, :, y0:y1, x0:x1] = x[idx, :, y0:y1, x0:x1]
    return mx, y, y[idx], 1 - (x1 - x0) * (y1 - y0) / (H * W)

# ============ Model ============
def build_model():
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(inf, 256), nn.BatchNorm1d(256),
        nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    return model.to(DEVICE)

# ============ Step 1: Re-split data ============
print(f"\n{'='*60}")
print("=== Step 1: Re-split with seed=123 ===")

# Gather all samples from old train/ and test/
all_samples = []
for cls in CLASS_NAMES:
    for subdir in ["train", "test"]:
        d = os.path.join(DATA_DIR, subdir, cls)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                    all_samples.append((os.path.join(d, f), cls))

print(f"Total available: {len(all_samples)}")
for cls in CLASS_NAMES:
    print(f"  {cls}: {sum(1 for _, c in all_samples if c == cls)}")

# Stratified split: 800/class train, 200/class test
paths = [s[0] for s in all_samples]
labels = [s[1] for s in all_samples]

train_paths, test_paths, train_labels, test_labels = train_test_split(
    paths, labels, test_size=800, stratify=labels, random_state=NEW_SEED
)
# Further split train into train+val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=int(0.15 * len(train_paths)),
    stratify=train_labels, random_state=NEW_SEED
)

print(f"\nNew split (seed={NEW_SEED}):")
print(f"  train: {len(train_paths)}")
for cls in CLASS_NAMES:
    print(f"    {cls}: {train_labels.count(cls)}")
print(f"  val:   {len(val_paths)}")
print(f"  test:  {len(test_paths)}")

# Create temporary directory structure for training
TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp_split123")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

for split_name, split_paths, split_labels in [
    ("train", train_paths, train_labels),
    ("val", val_paths, val_labels),
    ("test", test_paths, test_labels),
]:
    for cls in CLASS_NAMES:
        d = os.path.join(TMP_DIR, split_name, cls)
        os.makedirs(d, exist_ok=True)
    for p, l in zip(split_paths, split_labels):
        dst = os.path.join(TMP_DIR, split_name, l, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)

# ============ Step 2: Train ============
print(f"\n{'='*60}")
print("=== Step 2: Train with v8 recipe ===")

random.seed(NEW_SEED); np.random.seed(NEW_SEED)
torch.manual_seed(NEW_SEED); torch.cuda.manual_seed_all(NEW_SEED)

tr_ds = WeatherDS(os.path.join(TMP_DIR, "train"), train_tf)
vl_ds = WeatherDS(os.path.join(TMP_DIR, "val"), val_tf)
tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = build_model()
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

for n, p in model.named_parameters(): p.requires_grad = "classifier" in n
opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler() if USE_AMP else None

best_f1, best_ep, es_cnt = 0.0, 0, 0
ckpt_path = os.path.join(OUTPUT_DIR, f"validate_split{NEW_SEED}.pth")
t0 = time.time()

for ep in range(NUM_EPOCHS):
    if ep == WARMUP_EPOCHS:
        print(f"  >>> E{ep+1}: Unfreeze backbone")
        for p in model.parameters(): p.requires_grad = True
        hp = [p for n, p in model.named_parameters() if "classifier" in n]
        bp = [p for n, p in model.named_parameters() if "classifier" not in n]
        opt = optim.AdamW([{"params": hp, "lr": LR}, {"params": bp, "lr": FINE_LR}], weight_decay=WD)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

    model.train(); tl_ = 0.0
    for x, y in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_P
        if ma:
            if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup(x, y, MIXUP_A)
            else: x, ya, yb, lam = cutmix(x, y, CUTMIX_A)
        opt.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if ma else criterion(out, y)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            out = model(x)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if ma else criterion(out, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        tl_ += loss.item()
    sch.step()

    model.eval(); vp, vt = [], []
    with torch.no_grad():
        for x, y in vl:
            out = model(x.to(DEVICE))
            vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.cpu().numpy())
    acc = accuracy_score(vt, vp); f1 = f1_score(vt, vp, average="macro")
    lr = opt.param_groups[0]['lr']
    ph = "WARM" if ep < WARMUP_EPOCHS else "FT  "
    print(f"  {ph} E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(tl):.4f} | acc={acc:.4f} | f1={f1:.4f} | lr={lr:.2e}")
    if f1 > best_f1:
        best_f1, best_ep, es_cnt = f1, ep, 0
        torch.save({"epoch": ep, "model": model.state_dict(), "f1": f1, "acc": acc}, ckpt_path)
        print(f"    >>> Best (F1={best_f1:.4f})")
    else:
        es_cnt += 1
        if es_cnt >= EARLY_PATIENCE:
            print(f"    >>> EarlyStop @ E{ep+1}, Best F1={best_f1:.4f} @ E{best_ep+1}")
            break

elapsed = time.time() - t0
print(f"Train done: {elapsed:.0f}s ({elapsed/60:.1f}min) | Best Val F1={best_f1:.4f} @ E{best_ep+1}")

# ============ Step 3: TTA Test (new test set) ============
print(f"\n{'='*60}")
print("=== Step 3: TTA Test on NEW test set ===")

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()

test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(TMP_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d, f), CLASS_NAMES.index(cls)))

# TTA
preds_tta, tgts = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_predict(model, img)
    preds_tta.append(prob.argmax().item()); tgts.append(l)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_tta = accuracy_score(tgts, preds_tta)
f1_tta = f1_score(tgts, preds_tta, average="macro")
per_tta = f1_score(tgts, preds_tta, average=None)
cm_tta = confusion_matrix(tgts, preds_tta)

print(f"\nTTA Test (seed={NEW_SEED}):")
print(f"  Acc={acc_tta:.4f} F1={f1_tta:.4f}")
print(f"  cloudy={per_tta[0]:.4f} rain={per_tta[1]:.4f} sunny={per_tta[2]:.4f} snow={per_tta[3]:.4f}")
print(f"  cloudy->sunny={cm_tta[0][2]} sunny->cloudy={cm_tta[2][0]}")

# No-TTA
preds_no = []
for p, l in test_files:
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): preds_no.append(model(t).argmax().item())
f1_no = f1_score(tgts, preds_no, average="macro")
print(f"\nNo-TTA: F1={f1_no:.4f}")

# ============ Step 4: Compare ============
print(f"\n{'='*60}")
print("=== Step 4: Cross-Split Comparison ===")
print(f"{'':<20} {'Seed=42 (orig)':<16} {'Seed=123 (new)':<16} {'Diff':<10}")
print("-"*62)

orig_test_f1s = [0.8486, 0.8365, 0.8413, 0.8233, 0.8348]  # 5-fold single-model TTA
orig_test_mean = np.mean(orig_test_f1s)

print(f"  {'Test F1 (TTA)':<20} {orig_test_mean:.4f} (mean)     {f1_tta:.4f}             {f1_tta - orig_test_mean:+.4f}")

orig_f1_best = 0.8486  # Fold1
print(f"  {'Test F1 (best)':<20} {orig_f1_best:.4f}              {f1_tta:.4f}             {f1_tta - orig_f1_best:+.4f}")

print(f"\nPer-class comparison:")
print(f"  {'':<12} {'Orig (Fold1)':<14} {'New (seed=123)':<14}")
print(f"  {'cloudy':<12} {0.7385:<14.4f} {per_tta[0]:<14.4f}")
print(f"  {'rain':<12} {0.8900:<14.4f} {per_tta[1]:<14.4f}")
print(f"  {'sunny':<12} {0.8224:<14.4f} {per_tta[2]:<14.4f}")
print(f"  {'snow':<12} {0.9436:<14.4f} {per_tta[3]:<14.4f}")

# Final verdict
print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

diff = f1_tta - orig_test_mean
if abs(diff) < 0.015:
    print(f"[OK] New split F1 ({f1_tta:.4f}) vs orig mean ({orig_test_mean:.4f}): diff={diff:+.4f}")
    print("  v8 improvements are REAL. Model generalizes to new split.")
elif diff < -0.02:
    print(f"[!!] New split F1 ({f1_tta:.4f}) << orig mean ({orig_test_mean:.4f}): diff={diff:+.4f}")
    print("  Possible overfitting to original split. v8 gains may be inflated.")
else:
    print(f"[~] New split F1 ({f1_tta:.4f}) vs orig mean ({orig_test_mean:.4f}): diff={diff:+.4f}")
    print("  Slight difference, within normal variance.")

# Cleanup temp files (keep model checkpoint)
shutil.rmtree(TMP_DIR)
print(f"\nModel: {ckpt_path}")
print(f"Log: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
