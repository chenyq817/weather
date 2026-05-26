"""
从 SAM best checkpoint 恢复训练 — 换 AdamW 继续
加载 best_model_v12_sam.pth, 剩余 ~14 epoch, ~10min
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
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"resume_v12_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout; self.log = open(p, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

IMAGE_SIZE = 260; BATCH_SIZE = 32
LR = 5e-4; FINE_LR = 2e-5; WD = 3e-4
LABEL_SMOOTH = 0.1; MIXUP_A = 0.3; CUTMIX_A = 0.2; MIXUP_P = 0.7
GRAD_CLIP = 1.0
RESUME_EPOCH = 26  # approximate epoch we stopped at
TOTAL_EPOCHS = 40
EARLY_PATIENCE = 8
SEED = 42
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")

print(f"Resume from SAM best checkpoint, epoch {RESUME_EPOCH}, AdamW optimizer")
print(f"LR={LR} head, {FINE_LR} backbone")

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

# Load model
print("\nLoading SAM checkpoint...")
model = models.convnext_tiny(weights=None)
inf = model.classifier[2].in_features
model.classifier = nn.Sequential(
    nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
    nn.Dropout(0.4), nn.Linear(inf, 256),
    nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
)
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model_v12_sam.pth"), map_location="cpu")
model.load_state_dict(ckpt["model"])
model = model.to(DEVICE)
print(f"Loaded epoch={ckpt['epoch']+1}, val_f1={ckpt['f1']:.4f}")

# Data
ft = WeatherDS(os.path.join(DATA_DIR, "train"), train_tf)
from torch.utils.data import random_split
vs_ = max(1, int(0.15 * len(ft))); ts_ = len(ft) - vs_
tr_ds, _ = random_split(ft, [ts_, vs_], generator=torch.Generator().manual_seed(SEED))
fv = WeatherDS(os.path.join(DATA_DIR, "train"), val_tf)
_, vl_ds = random_split(fv, [ts_, vs_], generator=torch.Generator().manual_seed(SEED))
tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Optimizer: AdamW (not SAM), backbone already fine-tuned
for p in model.parameters(): p.requires_grad = True
hp = [p for n,p in model.named_parameters() if "classifier" in n]
bp = [p for n,p in model.named_parameters() if "classifier" not in n]
opt = optim.AdamW([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}], weight_decay=WD)
# Scheduler: start from where we left off (Cycle 3, ~epoch 8 of 20)
# CosineAnnealingLR with T_max set to remaining epochs
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

print(f"Training epochs {RESUME_EPOCH+1}-{TOTAL_EPOCHS}...")
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
best_f1, best_ep, es_cnt = ckpt["f1"], ckpt["epoch"], 0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v12_resumed.pth")
scaler = torch.amp.GradScaler()
t0 = time.time()

random.seed(SEED + RESUME_EPOCH); np.random.seed(SEED + RESUME_EPOCH)
torch.manual_seed(SEED + RESUME_EPOCH); torch.cuda.manual_seed_all(SEED + RESUME_EPOCH)

for ep in range(RESUME_EPOCH, TOTAL_EPOCHS):
    model.train(); tl_ = 0.0
    for x, y in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ma = torch.rand(1).item() < MIXUP_P
        if ma:
            if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup(x, y, MIXUP_A)
            else: x, ya, yb, lam = cutmix(x, y, CUTMIX_A)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
        scaler.scale(loss).backward()
        if GRAD_CLIP > 0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()
        tl_ += loss.item()
    sch.step()

    model.eval(); vp, vt = [], []
    with torch.no_grad():
        for x, y in vl:
            out = model(x.to(DEVICE))
            vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.cpu().numpy())
    acc = accuracy_score(vt, vp); f1 = f1_score(vt, vp, average="macro")
    lr = opt.param_groups[0]['lr']
    print(f"FT E{ep+1:3d}/{TOTAL_EPOCHS} | loss={tl_/len(tl):.4f} | acc={acc:.4f} | f1={f1:.4f} | lr={lr:.2e}")
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

# TTA Test
print(f"\n{'='*60}\n=== TTA Test ===")
ckpt_res = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt_res["model"]); model.eval()
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
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_f1[i]:.4f}")
print(f"  c->s={cm[0][2]} s->c={cm[2][0]}")

print(f"\nModel: {best_ckpt}\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
