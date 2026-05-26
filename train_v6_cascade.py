"""
v6: Cascade双头 + 针对性MixUp(cloudy<->sunny) + TTA推理
策略:
  1. 双头模型: Head_binary(cloudy vs not) + Head_3class(rain/sunny/snow)
  2. 针对性MixUp: 每个batch额外强制cloudy↔sunny混合
  3. TTA推理: 5crop+flip集成
  4. 推理时: binary头判cloudy → 用binary结果; 否则 → 3class头
目标: 打破cloudy↔sunny的混淆瓶颈
"""

import os, sys, time, datetime, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v6_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.t = sys.stdout; self.f = open(p, "w", encoding="utf-8")
    def write(self, m):
        self.t.write(m); self.f.write(m); self.f.flush()
    def flush(self):
        self.t.flush(); self.f.flush()

sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

# ============ 超参数 ============
IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS = 260, 64, 40
WARMUP_EPOCHS = 3
LR, FINE_LR, WD = 1e-3, 5e-5, 3e-4
LABEL_SMOOTH, MIXUP_A, CUTMIX_A, MIXUP_P = 0.1, 0.3, 0.2, 0.7
TARGET_MIXUP_P = 0.5  # 针对性cloudy↔sunny混合概率
GRAD_CLIP = 1.0
EARLY_PATIENCE = 10

CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
CLOUDY_IDX = 0
DEVICE = torch.device("cuda")
USE_AMP = True
TTA_CROPS = 5  # TTA推理crop数

print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)}) | {IMAGE_SIZE}px | Batch={BATCH_SIZE}")
print(f"Cascade + TargetedMixUp(cloudy<->sunny, p={TARGET_MIXUP_P}) + TTA(crops={TTA_CROPS})")
print(f"EarlyStop={EARLY_PATIENCE} | AMP={USE_AMP}")

# ============ 数据增强 ============
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.35, 0.35, 0.35, 0.1),
    transforms.RandomEqualize(p=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TTA变换: 5 crop + 水平翻转
def get_tta_transforms():
    base = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tflist = [base]
    # 四角crop
    for i in range(4):
        tflist.append(transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.FiveCrop(IMAGE_SIZE),
            transforms.Lambda(lambda crops, i=i: crops[i]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    return tflist

# ============ 数据集 ============
class WeatherDS(Dataset):
    def __init__(self, root, transform=None, return_idx=False):
        self.root = root; self.tf = transform; self.ret_idx = return_idx
        self.samples = []
        self.cls2idx = {n:i for i,n in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d,f), self.cls2idx[cls], cls))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l, cls = self.samples[idx]
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        if self.tf: img = self.tf(img)
        if self.ret_idx: return img, l, cls, p
        return img, l, cls

# ============ 模型: Cascade双头 ============
class CascadeModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        feat_dim = 1408

        # Head_binary: cloudy vs not_cloudy
        self.head_binary = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(feat_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 2),
        )

        # Head_3class: rain vs sunny vs snow
        self.head_3class = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(feat_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 3),
        )

    def forward(self, x, mode='both'):
        f = self.avgpool(self.features(x)).mean(dim=[2,3])
        if mode == 'binary': return self.head_binary(f)
        if mode == '3class': return self.head_3class(f)
        return self.head_binary(f), self.head_3class(f)

# ============ 数据加载 ============
print("\n=== Loading Data ===")
ft = WeatherDS(os.path.join(DATA_DIR, "train"), train_tf)
ts_ds = WeatherDS(os.path.join(DATA_DIR, "test"), val_tf)

for cls in CLASS_NAMES:
    print(f"  train/{cls}: {sum(1 for _,_,c in ft.samples if c==cls)}")
print(f"  test: {len(ts_ds)}")

from torch.utils.data import random_split
vs_ = max(1, int(0.15 * len(ft)))
ts_ = len(ft) - vs_
tr_ds, _ = random_split(ft, [ts_, vs_], generator=torch.Generator().manual_seed(42))
fv_ds = WeatherDS(os.path.join(DATA_DIR, "train"), val_tf)
_, vl_ds = random_split(fv_ds, [ts_, vs_], generator=torch.Generator().manual_seed(42))
print(f"train={ts_} val={vs_} test={len(ts_ds)}")

tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
tl_ts = DataLoader(ts_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============ 针对性 MixUp / CutMix ============
def mixup_tensor(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix_tensor(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, _, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W*np.sqrt(1-lam)), int(H*np.sqrt(1-lam))
    x0, y0 = np.clip(cx-bw//2, 0, W), np.clip(cy-bh//2, 0, H)
    x1, y1 = np.clip(cx+bw//2, 0, W), np.clip(cy+bh//2, 0, H)
    mx = x.clone(); mx[:,:,y0:y1,x0:x1] = x[idx,:,y0:y1,x0:x1]
    return mx, y, y[idx], 1 - (x1-x0)*(y1-y0)/(H*W)

def targeted_mixup_cloudy_sunny(x, y, cls_names, alpha=0.4):
    """针对性MixUp: 强制cloudy与sunny样本混合"""
    cloudy_mask = torch.tensor([n=="cloudy" for n in cls_names], device=x.device)
    sunny_mask = torch.tensor([n=="sunny" for n in cls_names], device=x.device)
    c_idx = torch.where(cloudy_mask)[0]
    s_idx = torch.where(sunny_mask)[0]

    if len(c_idx) < 2 or len(s_idx) < 2:
        return x, y, y, 1.0  # 不足，回退

    # 随机配对
    n_pair = min(len(c_idx), len(s_idx))
    c_sel = c_idx[torch.randperm(len(c_idx), device=x.device)[:n_pair]]
    s_sel = s_idx[torch.randperm(len(s_idx), device=x.device)[:n_pair]]

    lam = np.random.beta(alpha, alpha)
    mx = x.clone()
    mx[c_sel] = lam * x[c_sel] + (1 - lam) * x[s_sel]
    return mx, y[c_sel], y[s_sel], lam

# ============ 训练 ============
print("\n=== Building Cascade Model ===")
model = CascadeModel().to(DEVICE)
tp = sum(p.numel() for p in model.parameters())/1e6
print(f"Params: {tp:.2f}M | Dual head: binary(2) + 3class(3)")

crit_binary = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
crit_3class = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

# Phase 1: freeze backbone
for n, p in model.named_parameters():
    p.requires_grad = "features" not in n

opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler() if USE_AMP else None

best_f1, best_ep = 0.0, 0
es_counter = 0
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v6.pth")
t0 = time.time()

for ep in range(NUM_EPOCHS):
    if ep == WARMUP_EPOCHS:
        print(f"\n>>> Epoch {ep+1}: Unfreeze backbone")
        for p in model.parameters(): p.requires_grad = True
        hp = [p for n,p in model.named_parameters() if "features" not in n]
        bp = [p for n,p in model.named_parameters() if "features" in n]
        opt = optim.AdamW([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}], weight_decay=WD)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

    model.train()
    tl_ = 0.0

    for x, y, cls_names in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # === MixUp / CutMix ===
        ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_P
        if ma:
            if torch.rand(1).item() < 0.5:
                x, ya, yb, lam = mixup_tensor(x, y, MIXUP_A)
            else:
                x, ya, yb, lam = cutmix_tensor(x, y, CUTMIX_A)
            mix_active = True
        else:
            mix_active = False

        # === 针对性cloudy↔sunny混合 ===
        target_active = (ep >= WARMUP_EPOCHS and torch.rand(1).item() < TARGET_MIXUP_P)
        if target_active:
            x_t, c_t, s_t, lam_t = targeted_mixup_cloudy_sunny(x, y, cls_names, alpha=0.4)
            if c_t is not s_t:
                target_active = True
            else:
                target_active = False

        opt.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                out_bin, out_3c = model(x)
                # Binary标签: 0=cloudy, 1=not_cloudy
                y_bin = (y != CLOUDY_IDX).long()
                # 3class标签: 去掉cloudy, rain→0 sunny→1 snow→2
                y_3c = y.clone()
                y_3c[y == 0] = -1  # cloudy标记
                y_3c[y == 1] = 0   # rain
                y_3c[y == 2] = 1   # sunny
                y_3c[y == 3] = 2   # snow

                if mix_active:
                    ya_bin = (ya != CLOUDY_IDX).long()
                    yb_bin = (yb != CLOUDY_IDX).long()
                    l_bin = lam * crit_binary(out_bin, ya_bin) + (1-lam) * crit_binary(out_bin, yb_bin)
                    ya_3c = ya.clone(); ya_3c[ya==0]=-1; ya_3c[ya==1]=0; ya_3c[ya==2]=1; ya_3c[ya==3]=2
                    yb_3c = yb.clone(); yb_3c[yb==0]=-1; yb_3c[yb==1]=0; yb_3c[yb==2]=1; yb_3c[yb==3]=2
                    # 只对非cloudy样本计算3class loss
                    mask_a, mask_b = ya_3c >= 0, yb_3c >= 0
                    l_3c = 0.0
                    if mask_a.any():
                        l_3c += lam * crit_3class(out_3c[mask_a], ya_3c[mask_a])
                    if mask_b.any():
                        l_3c += (1-lam) * crit_3class(out_3c[mask_b], yb_3c[mask_b])
                    loss = l_bin + l_3c
                else:
                    mask_3c = y_3c >= 0
                    l_bin = crit_binary(out_bin, y_bin)
                    l_3c = crit_3class(out_3c[mask_3c], y_3c[mask_3c]) if mask_3c.any() else 0.0
                    loss = l_bin + l_3c

                # 针对性loss: cloudy/sunny被强制混合后, binary头应困惑(两边的标签被混合)
                if target_active:
                    out_bin_t = model(x_t, mode='binary')
                    # 混合后binary输出应接近[0.5, 0.5]
                    l_target = 0.5 * (crit_binary(out_bin_t[c_t], (c_t != CLOUDY_IDX).long()) +
                                      crit_binary(out_bin_t[c_t], (s_t != CLOUDY_IDX).long()))
                    loss += 0.3 * l_target

            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            # No AMP fallback
            out_bin, out_3c = model(x)
            y_bin = (y != CLOUDY_IDX).long()
            y_3c = y.clone()
            y_3c[y==0]=-1; y_3c[y==1]=0; y_3c[y==2]=1; y_3c[y==3]=2
            if mix_active:
                ya_bin = (ya != CLOUDY_IDX).long(); yb_bin = (yb != CLOUDY_IDX).long()
                l_bin = lam*crit_binary(out_bin,ya_bin)+(1-lam)*crit_binary(out_bin,yb_bin)
                ya_3c = ya.clone(); ya_3c[ya==0]=-1; ya_3c[ya==1]=0; ya_3c[ya==2]=1; ya_3c[ya==3]=2
                yb_3c = yb.clone(); yb_3c[yb==0]=-1; yb_3c[yb==1]=0; yb_3c[yb==2]=1; yb_3c[yb==3]=2
                ma_a, ma_b = ya_3c>=0, yb_3c>=0
                l_3c = (lam*crit_3class(out_3c[ma_a],ya_3c[ma_a]) if ma_a.any() else 0) + \
                       ((1-lam)*crit_3class(out_3c[ma_b],yb_3c[ma_b]) if ma_b.any() else 0)
                loss = l_bin + l_3c
            else:
                m3 = y_3c >= 0
                l_bin = crit_binary(out_bin, y_bin)
                l_3c = crit_3class(out_3c[m3], y_3c[m3]) if m3.any() else 0.0
                loss = l_bin + l_3c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        tl_ += loss.item()

    sch.step()

    # === Validate (simple, no TTA) ===
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y, _ in vl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out_bin, out_3c = model(x)
            bin_prob = torch.softmax(out_bin, 1)
            # Cascade推理
            is_cloudy = bin_prob[:, 0] > 0.7  # high confidence cloudy
            preds = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
            preds[is_cloudy] = 0  # cloudy
            # not cloudy → 3class head
            if (~is_cloudy).any():
                out_3 = out_3c[~is_cloudy]
                # 3class输出映射: 0→rain(1), 1→sunny(2), 2→snow(3)
                pred_3 = out_3.argmax(1) + 1
                preds[~is_cloudy] = pred_3
            all_preds.extend(preds.cpu().numpy())
            all_tgts.extend(y.cpu().numpy())

    acc = accuracy_score(all_tgts, all_preds)
    f1 = f1_score(all_tgts, all_preds, average="macro")
    lr = opt.param_groups[0]['lr']
    ph = "WARM" if ep < WARMUP_EPOCHS else "FT  "
    print(f"{ph} E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(tl):.4f} | acc={acc:.4f} | f1={f1:.4f} | lr={lr:.2e}")

    if f1 > best_f1:
        best_f1, best_ep = f1, ep
        es_counter = 0
        torch.save({"epoch":ep, "model":model.state_dict(), "f1":f1, "acc":acc}, best_ckpt)
        print(f"  >>> Best (F1={best_f1:.4f})")
    else:
        es_counter += 1
        if es_counter >= EARLY_PATIENCE:
            print(f"\n>>> EarlyStop @ Epoch {ep+1}, Best F1={best_f1:.4f}")
            break

print(f"\nDone: {time.time()-t0:.0f}s | Best val F1={best_f1:.4f} @ E{best_ep+1}")

# ============ TTA 测试评估 ============
print("\n=== TTA Test Evaluation ===")
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

tta_tfs = get_tta_transforms()
test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d, f), CLASS_NAMES.index(cls)))

print(f"Test images: {len(test_files)}")

all_preds, all_tgts = [], []
t0 = time.time()

for img_path, label in test_files:
    try: img = Image.open(img_path).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))

    # TTA: 多个变换取平均
    bin_logits, three_logits = [], []
    for tf_fn in tta_tfs:
        inp = tf_fn(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ob, o3 = model(inp)
            bin_logits.append(torch.softmax(ob, 1))
            three_logits.append(torch.softmax(o3, 1))

    # 平均概率
    bin_prob = torch.stack(bin_logits).mean(0)
    three_prob = torch.stack(three_logits).mean(0)

    # Cascade推理
    if bin_prob[0, 0] > 0.7:
        pred = 0  # cloudy
    else:
        pred = three_prob.argmax(1).item() + 1  # → rain(1), sunny(2), snow(3)

    all_preds.append(pred)
    all_tgts.append(label)

tt = time.time() - t0
acc = accuracy_score(all_tgts, all_preds)
f1m = f1_score(all_tgts, all_preds, average="macro")
per_f1 = f1_score(all_tgts, all_preds, average=None)

print(f"Accuracy: {acc:.4f} | F1 macro: {f1m:.4f} | Time: {tt:.1f}s ({tt/len(test_files)*1000:.1f}ms/img)")
print(f"\n" + classification_report(all_tgts, all_preds, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(all_tgts, all_preds)
print("Confusion Matrix:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(4)))

# ============ 对比 ============
print(f"\n=== All Versions ===")
base = {"cloudy":0.6801,"rain":0.8550,"sunny":0.7740,"snow":0.9147,"macro":0.8060}
v3  = {"cloudy":0.6981,"rain":0.8432,"sunny":0.7889,"snow":0.9254,"macro":0.8139}
v5  = {"cloudy":0.7053,"rain":0.8676,"sunny":0.8104,"snow":0.9436,"macro":0.8317}
print(f"{'':<12} {'Base':<10} {'v3(B2)':<10} {'v5(GPU)':<10} {'v6(Cas)':<10}")
print("-"*52)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<10} {base[cls]:.4f}     {v3[cls]:.4f}    {v5[cls]:.4f}    {per_f1[i]:.4f}")
print(f"  {'macro':<10} {base['macro']:.4f}     {v3['macro']:.4f}    {v5['macro']:.4f}    {f1m:.4f}")

print(f"\nCloudy<->Sunny confusion:")
print(f"  Base: c->s=44 s->c=32")
print(f"  v5:   c->s=41 s->c=25")
print(f"  v6:   c->s={cm[0][2]} s->c={cm[2][0]}")

# 统计cascade效果
correct_as_cloudy = cm[0][0]  # cloudy正确
total_pred_cloudy = cm[0].sum()
print(f"\nCascade binary head: cloudy recall={correct_as_cloudy}/{total_pred_cloudy} ({correct_as_cloudy/total_pred_cloudy*100:.1f}%)")

print(f"\nModel: {best_ckpt}")
print(f"Log: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
