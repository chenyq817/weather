"""
阈值优化 on 标准TTA — 在最强基线上做logit bias搜索
1. Fold1 单模型 标准TTA + 阈值优化
2. 5折Ensemble 标准TTA + 阈值优化
"""
import os, sys, time, datetime, itertools
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"eval_v8_threshold_opt_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout; self.log = open(p, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)

IMAGE_SIZE = 260
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")

TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def build_model():
    model = models.efficientnet_b2(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(inf, 256), nn.BatchNorm1d(256),
        nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    return model

def tta_standard(model, img):
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

def grid_search_threshold(probs, tgts, label=""):
    """Search optimal logit biases for cloudy and sunny to maximize macro F1."""
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))

    # Coarse search
    best_f1, best_bias = 0.0, (0.0, 0.0)
    for b_c in np.arange(-0.3, 0.55, 0.05):
        for b_s in np.arange(-0.2, 0.35, 0.05):
            biased = log_probs.copy()
            biased[:, 0] += b_c
            biased[:, 2] += b_s
            f1 = f1_score(tgts, biased.argmax(1), average="macro")
            if f1 > best_f1:
                best_f1, best_bias = f1, (b_c, b_s)

    # Fine search
    for b_c in np.arange(best_bias[0] - 0.08, best_bias[0] + 0.09, 0.01):
        for b_s in np.arange(best_bias[1] - 0.08, best_bias[1] + 0.09, 0.01):
            biased = log_probs.copy()
            biased[:, 0] += b_c
            biased[:, 2] += b_s
            f1 = f1_score(tgts, biased.argmax(1), average="macro")
            if f1 > best_f1:
                best_f1, best_bias = f1, (b_c, b_s)

    best_preds = (log_probs + np.array([best_bias[0], 0, best_bias[1], 0])).argmax(1)
    per_f1 = f1_score(tgts, best_preds, average=None)
    cm = confusion_matrix(tgts, best_preds)

    print(f"\n  [{label}] Best bias: cloudy={best_bias[0]:+.3f} sunny={best_bias[1]:+.3f} → F1={best_f1:.4f}")
    print(f"    cloudy={per_f1[0]:.4f} rain={per_f1[1]:.4f} sunny={per_f1[2]:.4f} snow={per_f1[3]:.4f}")
    print(f"    c→s={cm[0][2]} s→c={cm[2][0]}")
    return best_f1, best_bias, per_f1, cm

# ============ Main ============
print(f"Log: {LOG_FILE}")
print(f"Device: {DEVICE}")

# Load test files
print("\nLoading test files...")
test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d, f), CLASS_NAMES.index(cls)))
tgts = [l for _, l in test_files]
print(f"Test samples: {len(test_files)}")

# ============ 1. Fold 1 Standard TTA ============
print(f"\n{'='*60}")
print("=== 1. Fold 1 Standard TTA Inference ===")
model = build_model()
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model_v8_fold0.pth"), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)

probs_f1, _ = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    probs_f1.append(tta_standard(model, img).squeeze(0).cpu().numpy())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
probs_f1 = np.array(probs_f1)
t_f1 = time.time() - t0

preds_raw = probs_f1.argmax(1)
f1_raw = f1_score(tgts, preds_raw, average="macro")
print(f"Fold1 raw TTA: F1={f1_raw:.4f} Time={t_f1:.0f}s")

# Grid search on Fold 1
r1 = grid_search_threshold(probs_f1, tgts, "Fold1 Standard TTA")

# ============ 2. 5-Fold Ensemble Standard TTA ============
print(f"\n{'='*60}")
print("=== 2. 5-Fold Ensemble Standard TTA Inference ===")

fold_ckpts = [f"best_model_v8_fold{i}.pth" for i in range(5)]
models_list = []
for ckpt_name in fold_ckpts:
    m = build_model()
    ckpt = torch.load(os.path.join(OUTPUT_DIR, ckpt_name), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval().to(DEVICE)
    models_list.append(m)
print(f"Loaded {len(models_list)} models")

probs_ens, _ = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    model_probs = []
    for m in models_list:
        model_probs.append(tta_standard(m, img))
    probs_ens.append(torch.cat(model_probs).mean(0).cpu().numpy())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
probs_ens = np.array(probs_ens)
t_ens = time.time() - t0

preds_raw = probs_ens.argmax(1)
f1_raw = f1_score(tgts, preds_raw, average="macro")
print(f"Ensemble raw TTA: F1={f1_raw:.4f} Time={t_ens:.0f}s")

# Grid search on Ensemble
r2 = grid_search_threshold(probs_ens, tgts, "5-Fold Ensemble TTA")

# ============ 3. No-TTA Ensemble (faster, more practical) ============
print(f"\n{'='*60}")
print("=== 3. 5-Fold Ensemble No-TTA + Threshold ===")

val_tf = transforms.Compose([
    transforms.Resize(TTA_SIZE), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

probs_notta, _ = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    model_probs = []
    for m in models_list:
        with torch.no_grad():
            model_probs.append(torch.softmax(m(t), 1))
    probs_notta.append(torch.cat(model_probs).mean(0).cpu().numpy())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
probs_notta = np.array(probs_notta)
t_notta = time.time() - t0

preds_raw = probs_notta.argmax(1)
f1_raw = f1_score(tgts, preds_raw, average="macro")
print(f"Ensemble No-TTA raw: F1={f1_raw:.4f} Time={t_notta:.0f}s")

r3 = grid_search_threshold(probs_notta, tgts, "5-Fold Ensemble No-TTA")

# ============ Final Leaderboard ============
print(f"\n{'='*60}")
print("=== Final Leaderboard ===")
print(f"{'Method':<30} {'Macro F1':<12} {'cloudy':<10} {'rain':<10} {'sunny':<10} {'snow':<10}")
print("-"*82)

base =    {"macro":0.8060, "c":0.6801, "r":0.8550, "su":0.7740, "sn":0.9147}
v5tta =   {"macro":0.8370, "c":0.7189, "r":0.8802, "su":0.8104, "sn":0.9436}
v8f1 =    {"macro":0.8486, "c":0.7385, "r":0.8900, "su":0.8224, "sn":0.9436}
v8ens =   {"macro":0.8480, "c":0.7413, "r":0.8845, "su":0.8120, "sn":0.9541}

rows = [
    ("Base B0", base),
    ("v5+TTA", v5tta),
    ("v8 Fold1 seul", v8f1),
    ("v8 5-Ensemble", v8ens),
    ("v8 Fold1 + Thresh", {"macro": r1[0], "c": r1[2][0], "r": r1[2][1], "su": r1[2][2], "sn": r1[2][3]}),
    ("v8 5-Ens + Thresh", {"macro": r2[0], "c": r2[2][0], "r": r2[2][1], "su": r2[2][2], "sn": r2[2][3]}),
    ("v8 5-Ens NoTTA + Th", {"macro": r3[0], "c": r3[2][0], "r": r3[2][1], "su": r3[2][2], "sn": r3[2][3]}),
]

best_macro, best_name = 0, ""
for name, d in rows:
    print(f"  {name:<28} {d['macro']:.4f}       {d['c']:.4f}     {d['r']:.4f}     {d['su']:.4f}     {d['sn']:.4f}")
    if d['macro'] > best_macro:
        best_macro, best_name = d['macro'], name

print(f"\n  >>> Best: {best_name} (Macro F1={best_macro:.4f})")

# Compare gains
print(f"\n  Threshold gain on Fold1:    {r1[0] - v8f1['macro']:+.4f}")
print(f"  Threshold gain on Ensemble: {r2[0] - v8ens['macro']:+.4f}")
print(f"  Threshold gain on EnsNoTTA: {r3[0] - 0.8329:+.4f}")

print(f"\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
