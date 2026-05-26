"""
v10: ConvNeXt-Tiny — LayerNorm架构, 规避BN+小batch问题
ConvNeXt-Tiny: 28M params, 7x7 kernel, LayerNorm, GELU
vs B2: 9M params, 3x3 kernel, BatchNorm, SiLU
vs B4: 19M params, 3x3 kernel, BatchNorm, SiLU (BN+小batch=失败)

优势:
  - LayerNorm对batch size不敏感, bs=32稳定收敛
  - 大kernel(7x7)捕获天空/云层大范围纹理
  - 现代架构, ImageNet top1=82.1% (vs B2=80.1%)
"""
import os, sys, time, datetime, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v10_convnext_s456_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout; self.log = open(p, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

# ============ Config ============
IMAGE_SIZE = 260
BATCH_SIZE = 32
NUM_EPOCHS = 40
WARMUP_EPOCHS = 3
LR = 8e-4
FINE_LR = 3e-5
WD = 3e-4
LABEL_SMOOTH = 0.1
MIXUP_A, CUTMIX_A, MIXUP_P = 0.3, 0.2, 0.7
GRAD_CLIP = 1.0
EARLY_PATIENCE = 10
SEED = 456

CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")
USE_AMP = True

print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"Backbone: ConvNeXt-Tiny | {IMAGE_SIZE}px | Batch={BATCH_SIZE}")
print(f"Arch: LayerNorm + GELU + 7x7 kernel | ~28M params | Seed={SEED}")

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
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def tta_predict(model, img):
    img_resized = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_resized)
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            tensor = norm_t(to_tensor(c)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs.append(torch.softmax(model(tensor), 1))
    return torch.cat(probs).mean(0, keepdim=True)

# ============ Dataset ============
class WeatherDS(Dataset):
    def __init__(self, root, transform=None):
        self.root = root; self.tf = transform
        self.samples = []; self.c2i = {n:i for i,n in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d,f), self.c2i[cls]))
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
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    B, _, H, W = x.size(); idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw = int(W * np.sqrt(1 - lam)); bh = int(H * np.sqrt(1 - lam))
    x0, y0 = np.clip(cx - bw//2, 0, W), np.clip(cy - bh//2, 0, H)
    x1, y1 = np.clip(cx + bw//2, 0, W), np.clip(cy + bh//2, 0, H)
    mx = x.clone(); mx[:,:,y0:y1,x0:x1] = x[idx,:,y0:y1,x0:x1]
    return mx, y, y[idx], 1 - (x1-x0)*(y1-y0)/(H*W)

# ============ Model ============
print("\nBuilding ConvNeXt-Tiny...")
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
inf = model.classifier[2].in_features  # 768
model.classifier = nn.Sequential(
    nn.Flatten(1),
    nn.LayerNorm(inf, eps=1e-6),
    nn.Dropout(0.4),
    nn.Linear(inf, 256),
    nn.GELU(),
    nn.Dropout(0.25),
    nn.Linear(256, NUM_CLASSES),
)
model = model.to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ============ Data ============
print("\nLoading data...")
ft = WeatherDS(os.path.join(DATA_DIR, "train"), train_tf)
ts_ds = WeatherDS(os.path.join(DATA_DIR, "test"), val_tf)
for cls in CLASS_NAMES:
    print(f"  train/{cls}: {sum(1 for _,l in ft.samples if CLASS_NAMES[l]==cls)}")
print(f"  test: {len(ts_ds)}")

from torch.utils.data import random_split
vs_ = max(1, int(0.15 * len(ft))); ts_ = len(ft) - vs_
tr_ds, _ = random_split(ft, [ts_, vs_], generator=torch.Generator().manual_seed(SEED))
fv = WeatherDS(os.path.join(DATA_DIR, "train"), val_tf)
_, vl_ds = random_split(fv, [ts_, vs_], generator=torch.Generator().manual_seed(SEED))
print(f"train={ts_} val={vs_} test={len(ts_ds)}")

tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============ Train ============
print("\nTraining...")
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

for n, p in model.named_parameters(): p.requires_grad = "classifier" in n
opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler() if USE_AMP else None

best_f1, best_ep, es_cnt = 0.0, 0, 0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v10_convnext_s456.pth")
t0 = time.time()

for ep in range(NUM_EPOCHS):
    if ep == WARMUP_EPOCHS:
        print(f"\n>>> E{ep+1}: Unfreeze backbone")
        for p in model.parameters(): p.requires_grad = True
        hp = [p for n,p in model.named_parameters() if "classifier" in n]
        bp = [p for n,p in model.named_parameters() if "classifier" not in n]
        opt = optim.AdamW([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}], weight_decay=WD)
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
                loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            out = model(x)
            loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
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
    print(f"{ph} E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(tl):.4f} | acc={acc:.4f} | f1={f1:.4f} | lr={lr:.2e}")

    if f1 > best_f1:
        best_f1, best_ep, es_cnt = f1, ep, 0
        torch.save({"epoch":ep,"model":model.state_dict(),"f1":f1,"acc":acc}, best_ckpt)
        print(f"  >>> Best (F1={best_f1:.4f})")
    else:
        es_cnt += 1
        if es_cnt >= EARLY_PATIENCE:
            print(f"\n>>> EarlyStop @ E{ep+1}, Best F1={best_f1:.4f} @ E{best_ep+1}")
            break

total_t = time.time() - t0
print(f"\nDone: {total_t:.0f}s ({total_t/60:.1f}min) | Best F1={best_f1:.4f} @ E{best_ep+1}")

# ============ TTA Test ============
print(f"\n{'='*60}")
print("=== TTA Test ===")
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()

test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d,f), CLASS_NAMES.index(cls)))

preds, tgts = [], []
t_start = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_predict(model, img)
    preds.append(prob.argmax().item()); tgts.append(l)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

tt = time.time() - t_start
acc = accuracy_score(tgts, preds); f1m = f1_score(tgts, preds, average="macro")
per_f1 = f1_score(tgts, preds, average=None)
print(f"\nTTA: Acc={acc:.4f} F1={f1m:.4f} Time={tt:.0f}s")
print(f"\n{classification_report(tgts, preds, target_names=CLASS_NAMES, digits=4)}")
cm = confusion_matrix(tgts, preds)
print("Confusion:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(4)))

# No-TTA
print(f"\n=== No-TTA ===")
sp, st = [], []
for p, l in test_files:
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): sp.append(model(t).argmax().item())
    st.append(l)
print(f"No-TTA: Acc={accuracy_score(st, sp):.4f} F1={f1_score(st, sp, average='macro'):.4f}")

# ============ Compare ============
print(f"\n{'='*60}")
print("=== vs v8 (B2) ===")
v8_tta = {"cloudy":0.7385, "rain":0.8900, "sunny":0.8224, "snow":0.9436, "macro":0.8486}
v10_tta = {CLASS_NAMES[i]: per_f1[i] for i in range(4)}
v10_tta["macro"] = f1m
print(f"{'':<12} {'v8(B2)':<10} {'v10(CT)':<10} {'Diff':<10}")
print("-"*42)
for cls in CLASS_NAMES:
    print(f"  {cls:<10} {v8_tta[cls]:.4f}     {v10_tta[cls]:.4f}     {v10_tta[cls]-v8_tta[cls]:+.4f}")
print(f"  {'macro':<10} {v8_tta['macro']:.4f}     {v10_tta['macro']:.4f}     {v10_tta['macro']-v8_tta['macro']:+.4f}")

print(f"\nCloudy<->Sunny:")
print(f"  v8(B2): c->s={29} s->c={32}")
print(f"  v10(CT): c->s={cm[0][2]} s->c={cm[2][0]}")

print(f"\nModel: {best_ckpt}")
print(f"Log: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
