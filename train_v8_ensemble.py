"""
v8: StratifiedKFold (k=5) + 增强调优 + Ensemble
策略:
  A(CV): 5折分层交叉验证, 每折val=640张, 模型选择更可靠
  C(增强): 降低破坏cloudy↔sunny区分度的变换强度
  E(Ensemble): 5个fold模型 + TTA = 50次预测平均
"""
import os, sys, time, datetime, json
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_v8_ensemble_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout
        self.log = open(p, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)
print(f"Log: {LOG_FILE}")

# ============ Config ============
IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS = 260, 64, 40
WARMUP_EPOCHS = 3
LR, FINE_LR, WD = 1e-3, 5e-5, 3e-4
LABEL_SMOOTH, MIXUP_A, CUTMIX_A, MIXUP_P = 0.1, 0.3, 0.2, 0.7
GRAD_CLIP = 1.0
EARLY_PATIENCE = 10
N_FOLDS = 5
FOLD_SEEDS = [42, 123, 456, 789, 101112]
CV_RANDOM_STATE = 42

CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")
USE_AMP = True

print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)}) | {IMAGE_SIZE}px | Batch={BATCH_SIZE}")
print(f"Strategy: {N_FOLDS}-fold CV + Tuned Aug + Ensemble")
print(f"Fold seeds: {FOLD_SEEDS}")

# ============ Augmentation (Tuned for cloudy↔sunny) ============
# Key changes from v7:
#   ColorJitter reduced: preserve cloudy(dark)↔sunny(bright) brightness gap
#   RandomEqualize p↓:  histogram equalization destroys brightness distribution
#   RandAugment replaces TrivialAugmentWide: more controlled
#   Rotation reduced: 15° vs 25°, more realistic for weather photos
#   RandomAdjustSharpness: cloud texture(soft) vs sunny shadows(sharp)
#   RandomErasing scale↓: smaller erase, less likely to remove key sky features
#   Crop scale↑: less aggressive zoom-out, preserve global sky context

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

# ============ TTA helpers ============
TTA_SIZE = int(IMAGE_SIZE * 1.14)

def tta_predict(model, img, device):
    """5-crop + flip = 10 predictions → average"""
    img_resized = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_resized)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            tensor = norm(to_tensor(c)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs.append(torch.softmax(model(tensor), 1))
    return torch.cat(probs).mean(0, keepdim=True)

# ============ Dataset ============
class WeatherDS(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.tf = transform
        self.samples = []
        self.c2i = {n: i for i, n in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((os.path.join(d, f), self.c2i[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.tf:
            img = self.tf(img)
        return img, l

# ============ MixUp / CutMix ============
def mixup(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    B, _, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x0, y0 = np.clip(cx - bw // 2, 0, W), np.clip(cy - bh // 2, 0, H)
    x1, y1 = np.clip(cx + bw // 2, 0, W), np.clip(cy + bh // 2, 0, H)
    mx = x.clone()
    mx[:, :, y0:y1, x0:x1] = x[idx, :, y0:y1, x0:x1]
    return mx, y, y[idx], 1 - (x1 - x0) * (y1 - y0) / (H * W)

# ============ Model builder ============
def build_model():
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(inf, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(DEVICE)

# ============ Train one fold ============
def train_fold(fold_idx, train_indices, val_indices, full_dataset, seed):
    """Train a single fold, return best val F1 and model path."""
    print(f"\n{'='*60}")
    print(f"=== Fold {fold_idx+1}/{N_FOLDS} | Seed={seed}")
    print(f"=== Train={len(train_indices)} Val={len(val_indices)}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tr_ds = Subset(full_dataset, train_indices)
    vl_ds = Subset(WeatherDS(os.path.join(DATA_DIR, "train"), val_tf), val_indices)

    tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Build fresh model
    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Warmup: freeze backbone
    for n, p in model.named_parameters():
        p.requires_grad = "classifier" in n
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if USE_AMP else None

    best_f1, best_ep, es_cnt = 0.0, 0, 0
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_v8_fold{fold_idx}.pth")
    t0 = time.time()

    for ep in range(NUM_EPOCHS):
        if ep == WARMUP_EPOCHS:
            print(f"  >>> E{ep+1}: Unfreeze backbone")
            for p in model.parameters():
                p.requires_grad = True
            hp = [p for n, p in model.named_parameters() if "classifier" in n]
            bp = [p for n, p in model.named_parameters() if "classifier" not in n]
            opt = optim.AdamW([{"params": hp, "lr": LR}, {"params": bp, "lr": FINE_LR}], weight_decay=WD)
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

        model.train()
        tl_ = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_P
            if ma:
                if torch.rand(1).item() < 0.6:
                    x, ya, yb, lam = mixup(x, y, MIXUP_A)
                else:
                    x, ya, yb, lam = cutmix(x, y, CUTMIX_A)

            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if ma else criterion(out, y)
                scaler.scale(loss).backward()
                if GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if ma else criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            tl_ += loss.item()

        sch.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(DEVICE))
                vp.extend(out.argmax(1).cpu().numpy())
                vt.extend(y.cpu().numpy())

        acc = accuracy_score(vt, vp)
        f1 = f1_score(vt, vp, average="macro")
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
    print(f"  Fold {fold_idx+1} done: {elapsed:.0f}s ({elapsed/60:.1f}min) | Best Val F1={best_f1:.4f} @ E{best_ep+1}")
    return best_f1, ckpt_path

# ============ Ensemble inference on test set ============
def ensemble_test(model_paths, test_files):
    """Load all fold models, run TTA ensemble, return metrics."""
    models_list = []
    for mp in model_paths:
        model = build_model()
        ckpt = torch.load(mp, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models_list.append(model)
    print(f"\nLoaded {len(models_list)} models for ensemble")

    all_probs = []
    t_start = time.time()
    for i, (p, l) in enumerate(test_files):
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        # Collect all TTA probs from all models
        model_probs = []
        for model in models_list:
            model_probs.append(tta_predict(model, img, DEVICE))
        avg_prob = torch.cat(model_probs).mean(0, keepdim=True)
        all_probs.append(avg_prob)
        if (i + 1) % 200 == 0:
            print(f"  Ensemble TTA: {i+1}/{len(test_files)}")

    preds = [p.argmax().item() for p in all_probs]
    tgts = [l for _, l in test_files]
    elapsed = time.time() - t_start

    acc = accuracy_score(tgts, preds)
    f1m = f1_score(tgts, preds, average="macro")
    per_f1 = f1_score(tgts, preds, average=None)
    return acc, f1m, per_f1, preds, tgts, elapsed

def single_model_test(model_path, test_files, label=""):
    """Test a single model with TTA."""
    model = build_model()
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    t0 = time.time()
    for p, l in test_files:
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        prob = tta_predict(model, img, DEVICE)
        preds.append(prob.argmax().item())
        tgts.append(l)

    acc = accuracy_score(tgts, preds)
    f1m = f1_score(tgts, preds, average="macro")
    per_f1 = f1_score(tgts, preds, average=None)
    elapsed = time.time() - t0
    print(f"  [{label}] TTA Acc={acc:.4f} F1={f1m:.4f} ({elapsed:.0f}s)")
    return acc, f1m, per_f1

# ============ Main ============
if __name__ == "__main__":
    import random

    # Load full training dataset (for k-fold indexing)
    print("\nLoading full training dataset...")
    full_train_ds = WeatherDS(os.path.join(DATA_DIR, "train"), train_tf)
    full_labels = [l for _, l in full_train_ds.samples]
    print(f"Total train samples: {len(full_train_ds)}")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {full_labels.count(CLASS_NAMES.index(cls))}")

    # Load test files
    print("\nLoading test files...")
    test_files = []
    for cls in CLASS_NAMES:
        d = os.path.join(DATA_DIR, "test", cls)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    test_files.append((os.path.join(d, f), CLASS_NAMES.index(cls)))
    print(f"Test samples: {len(test_files)}")

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_results = []  # list of (val_f1, ckpt_path)
    fold_test_results = []  # per-fold test metrics

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_train_ds)), full_labels)):
        seed = FOLD_SEEDS[fold_idx]
        val_f1, ckpt_path = train_fold(fold_idx, train_idx, val_idx, full_train_ds, seed)
        fold_results.append((val_f1, ckpt_path))

        # Quick single-model TTA test
        print(f"\n  --- Fold {fold_idx+1} Single-Model TTA Test ---")
        acc, f1m, per_f1 = single_model_test(ckpt_path, test_files, f"Fold{fold_idx+1}")
        fold_test_results.append({"fold": fold_idx, "val_f1": val_f1, "test_f1": f1m,
                                   "test_acc": acc, "per_f1": per_f1.tolist(), "ckpt": ckpt_path})

    # ============ CV Summary ============
    val_f1s = [r[0] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"=== {N_FOLDS}-Fold CV Summary ===")
    print(f"  Val F1 per fold: {[f'{v:.4f}' for v in val_f1s]}")
    print(f"  CV Mean: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"  Test F1 per fold: {[f'{r['test_f1']:.4f}' for r in fold_test_results]}")
    print(f"  CV Test Mean: {np.mean([r['test_f1'] for r in fold_test_results]):.4f}")

    # ============ Ensemble Test ============
    print(f"\n{'='*60}")
    print(f"=== Ensemble TTA Test ({N_FOLDS} models × 10 preds = {N_FOLDS*10} preds/img) ===")
    all_ckpts = [r[1] for r in fold_results]
    ens_acc, ens_f1, ens_per_f1, ens_preds, ens_tgts, ens_time = ensemble_test(all_ckpts, test_files)

    print(f"\nEnsemble TTA: Acc={ens_acc:.4f} | F1={ens_f1:.4f} | Time={ens_time:.0f}s")
    print(f"\n{classification_report(ens_tgts, ens_preds, target_names=CLASS_NAMES, digits=4)}")

    cm = confusion_matrix(ens_tgts, ens_preds)
    print("Ensemble Confusion:")
    print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(4)))

    # ============ No-TTA Ensemble ============
    print(f"\n=== Ensemble No-TTA Test ===")
    models_list = []
    for mp in all_ckpts:
        model = build_model()
        ckpt = torch.load(mp, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models_list.append(model)

    no_tta_preds, no_tta_tgts = [], []
    for p, l in test_files:
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        t = val_tf(img).unsqueeze(0).to(DEVICE)
        model_probs = []
        for model in models_list:
            with torch.no_grad():
                model_probs.append(torch.softmax(model(t), 1))
        no_tta_preds.append(torch.cat(model_probs).mean(0).argmax().item())
        no_tta_tgts.append(l)

    no_tta_acc = accuracy_score(no_tta_tgts, no_tta_preds)
    no_tta_f1 = f1_score(no_tta_tgts, no_tta_preds, average="macro")
    print(f"No-TTA Ensemble: Acc={no_tta_acc:.4f} F1={no_tta_f1:.4f}")
    print(f"TTA Gain: {ens_f1 - no_tta_f1:+.4f}")

    # ============ Final Leaderboard ============
    print(f"\n{'='*60}")
    print("=== Final Leaderboard ===")
    print(f"{'':<12} {'Base':<10} {'v5':<10} {'v7(TTA)':<10} {'v8-Best':<10} {'v8-Ens':<10}")
    print("-"*62)

    base = {"cloudy": 0.6801, "rain": 0.8550, "sunny": 0.7740, "snow": 0.9147, "macro": 0.8060}
    v5 = {"cloudy": 0.7053, "rain": 0.8676, "sunny": 0.8104, "snow": 0.9436, "macro": 0.8317}
    v7 = {"cloudy": 0.6897, "rain": 0.8802, "sunny": 0.7916, "snow": 0.9354, "macro": 0.8242}

    # Best single fold
    best_fold = max(fold_test_results, key=lambda r: r["test_f1"])
    v8_best = {name: best_fold["per_f1"][i] for i, name in enumerate(CLASS_NAMES)}
    v8_best["macro"] = best_fold["test_f1"]

    v8_ens = {name: ens_per_f1[i] for i, name in enumerate(CLASS_NAMES)}
    v8_ens["macro"] = ens_f1

    for cls in CLASS_NAMES:
        print(f"  {cls:<10} {base[cls]:.4f}     {v5[cls]:.4f}     {v7[cls]:.4f}     {v8_best[cls]:.4f}     {v8_ens[cls]:.4f}")
    print(f"  {'macro':<10} {base['macro']:.4f}     {v5['macro']:.4f}     {v7['macro']:.4f}     {v8_best['macro']:.4f}     {v8_ens['macro']:.4f}")

    print(f"\nCloudy<->Sunny confusion:")
    print(f"  Base:  c→s=44 s→c=32")
    print(f"  v5:    c→s=41 s→c=25")
    print(f"  v7:    c→s=48 s→c=24")
    print(f"  v8Ens: c→s={cm[0][2]} s→c={cm[2][0]}")

    # ============ Save results ============
    results = {
        "config": {
            "n_folds": N_FOLDS, "fold_seeds": FOLD_SEEDS,
            "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "augmentation": "tuned v8 (reduced jitter/equalize/rotation, RandAugment, AdjustSharpness)",
        },
        "cv": {
            "val_f1s": val_f1s,
            "cv_mean": float(np.mean(val_f1s)),
            "cv_std": float(np.std(val_f1s)),
        },
        "folds": [{"fold": r["fold"], "val_f1": r["val_f1"], "test_f1": r["test_f1"],
                    "per_f1": r["per_f1"]} for r in fold_test_results],
        "ensemble": {
            "tta_f1": float(ens_f1), "tta_acc": float(ens_acc),
            "no_tta_f1": float(no_tta_f1), "no_tta_acc": float(no_tta_acc),
            "tta_gain": float(ens_f1 - no_tta_f1),
            "per_f1": ens_per_f1.tolist(),
            "confusion": cm.tolist(),
        },
        "comparison": {
            "base": base, "v5": v5, "v7": v7,
            "v8_best_fold": v8_best, "v8_ensemble": v8_ens,
        },
    }
    results_path = os.path.join(OUTPUT_DIR, "results_v8_ensemble.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    print(f"\nModels: {[r[1] for r in fold_results]}")
    print(f"Log: {LOG_FILE}")
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
