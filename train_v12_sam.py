"""
v12: ConvNeXt-Tiny + SAM (Sharpness-Aware Minimization)
SAM: 寻找loss平面最平坦的minima → 更好泛化
训练时间 ×2 (~30min), 预期 +0.3~0.8%
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
LOG_FILE = os.path.join(LOG_DIR, f"training_v12_sam_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout; self.log = open(p, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

# ============ SAM Optimizer ============
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm() + 1e-12
        for group in self.param_groups:
            scale = group["rho"] / grad_norm
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for group in self.param_groups
                        for p in group["params"] if p.grad is not None]), p=2)
        return norm

# ============ Config ============
IMAGE_SIZE = 260; BATCH_SIZE = 32; NUM_EPOCHS = 40; WARMUP_EPOCHS = 3
LR = 8e-4; FINE_LR = 3e-5; WD = 3e-4
LABEL_SMOOTH = 0.1; MIXUP_A = 0.3; CUTMIX_A = 0.2; MIXUP_P = 0.7
GRAD_CLIP = 1.0; EARLY_PATIENCE = 12; SEED = 42
SAM_RHO = 0.05

CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda"); USE_AMP = False  # AMP conflicts with SAM's dual step

print(f"Device: {DEVICE} | ConvNeXt-Tiny + SAM(rho={SAM_RHO}) | {IMAGE_SIZE}px | Batch={BATCH_SIZE}")

# ============ Augmentation ============
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomEqualize(p=0.1), transforms.RandomGrayscale(p=0.05),
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
inf = model.classifier[2].in_features
model.classifier = nn.Sequential(
    nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
    nn.Dropout(0.4), nn.Linear(inf, 256),
    nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
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
opt = SAM(model.parameters(), base_optimizer=optim.AdamW, rho=SAM_RHO, lr=LR, weight_decay=WD)
sch = optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler() if USE_AMP else None

best_f1, best_ep, es_cnt = 0.0, 0, 0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v12_sam.pth")
t0 = time.time()

for ep in range(NUM_EPOCHS):
    if ep == WARMUP_EPOCHS:
        print(f"\n>>> E{ep+1}: Unfreeze backbone")
        for p in model.parameters(): p.requires_grad = True
        hp = [p for n,p in model.named_parameters() if "classifier" in n]
        bp = [p for n,p in model.named_parameters() if "classifier" not in n]
        opt = SAM([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}],
                  base_optimizer=optim.AdamW, rho=SAM_RHO, weight_decay=WD)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt.base_optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    model.train(); tl_ = 0.0
    for x, y in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_P
        if ma:
            if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup(x, y, MIXUP_A)
            else: x, ya, yb, lam = cutmix(x, y, CUTMIX_A)

        # ---- SAM first pass ----
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss1 = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            scaler.scale(loss1).backward()
            scaler.unscale_(opt)
        else:
            out = model(x)
            loss1 = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            loss1.backward()

        if GRAD_CLIP > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.first_step(zero_grad=True)

        # ---- SAM second pass (at perturbed weights) ----
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss2 = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            scaler.scale(loss2).backward()
            scaler.unscale_(opt)
        else:
            out = model(x)
            loss2 = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
            loss2.backward()

        if GRAD_CLIP > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.second_step(zero_grad=True)
        if USE_AMP: scaler.update()

        tl_ += loss2.item()

    sch.step()

    model.eval(); vp, vt = [], []
    with torch.no_grad():
        for x, y in vl:
            out = model(x.to(DEVICE))
            vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.cpu().numpy())
    acc = accuracy_score(vt, vp); f1 = f1_score(vt, vp, average="macro")
    lr = opt.base_optimizer.param_groups[0]['lr']
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
print(f"\n{'='*60}\n=== TTA Test ===")
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
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_predict(model, img)
    preds.append(prob.argmax().item()); tgts.append(l)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
acc = accuracy_score(tgts, preds); f1m = f1_score(tgts, preds, average="macro")
per_f1 = f1_score(tgts, preds, average=None); cm = confusion_matrix(tgts, preds)
print(f"\nTTA: Acc={acc:.4f} F1={f1m:.4f}")
print(f"\n{classification_report(tgts, preds, target_names=CLASS_NAMES, digits=4)}")
print("Confusion:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(4)))
print(f"\n=== No-TTA ==="); sp, st = [], []
for p, l in test_files:
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): sp.append(model(t).argmax().item()); st.append(l)
print(f"No-TTA: Acc={accuracy_score(st, sp):.4f} F1={f1_score(st, sp, average='macro'):.4f}")

# ============ Compare ============
print(f"\n{'='*60}\n=== vs v10 (CT plain) ===")
v10_tta = {"cloudy":0.7896, "rain":0.9157, "sunny":0.8465, "snow":0.9646, "macro":0.8791}
v12_tta = {CLASS_NAMES[i]: per_f1[i] for i in range(4)}; v12_tta["macro"] = f1m
print(f"{'':<12} {'v10(AdamW)':<12} {'v12(SAM)':<12} {'Diff':<10}")
print("-"*46)
for cls in CLASS_NAMES:
    print(f"  {cls:<10} {v10_tta[cls]:.4f}       {v12_tta[cls]:.4f}       {v12_tta[cls]-v10_tta[cls]:+.4f}")
print(f"  {'macro':<10} {v10_tta['macro']:.4f}       {v12_tta['macro']:.4f}       {v12_tta['macro']-v10_tta['macro']:+.4f}")
print(f"\nCloudy<->Sunny: v10(CT): c->s=28 s->c=22")
print(f"                 v12(SAM): c->s={cm[0][2]} s->c={cm[2][0]}")
print(f"\nModel: {best_ckpt}\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
