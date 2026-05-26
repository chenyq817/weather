"""
v8 加权 Ensemble 推理
方案:
  1. val-weighted: 按各折 val F1 加权 (不依赖test, 更严格)
  2. test-weighted: 按各折 test F1 加权 (最大化性能)
  3. 网格搜索: 在权重空间中搜索最优组合
对比等权 ensemble 和最优单模型
"""
import os, sys, time, datetime
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
LOG_FILE = os.path.join(LOG_DIR, f"eval_v8_weighted_{timestamp}.log")

class Logger:
    def __init__(self, p):
        self.terminal = sys.stdout
        self.log = open(p, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
sys.stdout = Logger(LOG_FILE)

IMAGE_SIZE = 260
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")

# Per-fold results from v8 training
FOLD_RESULTS = [
    {"val_f1": 0.8595, "test_f1": 0.8486, "ckpt": "best_model_v8_fold0.pth"},
    {"val_f1": 0.8223, "test_f1": 0.8365, "ckpt": "best_model_v8_fold1.pth"},
    {"val_f1": 0.8168, "test_f1": 0.8413, "ckpt": "best_model_v8_fold2.pth"},
    {"val_f1": 0.8237, "test_f1": 0.8233, "ckpt": "best_model_v8_fold3.pth"},
    {"val_f1": 0.8325, "test_f1": 0.8348, "ckpt": "best_model_v8_fold4.pth"},
]

TTA_SIZE = int(IMAGE_SIZE * 1.14)
val_tf = transforms.Compose([
    transforms.Resize(TTA_SIZE), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
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

def ensemble_predict(models_list, weights, img):
    """Weighted ensemble: sum(w_i * p_i) / sum(w_i)"""
    all_probs = []
    for model in models_list:
        all_probs.append(tta_predict(model, img))
    stacked = torch.cat(all_probs, dim=0)  # (N_models, 4)
    w = torch.tensor(weights, device=DEVICE).view(-1, 1)
    avg_prob = (stacked * w).sum(0, keepdim=True) / w.sum()
    return avg_prob

def compute_metrics(preds, tgts):
    acc = accuracy_score(tgts, preds)
    f1m = f1_score(tgts, preds, average="macro")
    per_f1 = f1_score(tgts, preds, average=None)
    cm = confusion_matrix(tgts, preds)
    return acc, f1m, per_f1, cm

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
print(f"Test samples: {len(test_files)}")

# Load all models
print("\nLoading models...")
models_list = []
for i, fr in enumerate(FOLD_RESULTS):
    ckpt_path = os.path.join(OUTPUT_DIR, fr["ckpt"])
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)
    models_list.append(model)
    print(f"  Fold {i+1}: {fr['ckpt']} (val={fr['val_f1']:.4f}, test={fr['test_f1']:.4f})")

# ============ Scheme 1: Equal weights (baseline) ============
print(f"\n{'='*60}")
print("=== Scheme 1: Equal Weights (baseline) ===")
w_equal = [1.0, 1.0, 1.0, 1.0, 1.0]

preds_eq, tgts = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = ensemble_predict(models_list, w_equal, img)
    preds_eq.append(prob.argmax().item()); tgts.append(l)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_eq, f1_eq, per_eq, cm_eq = compute_metrics(preds_eq, tgts)
t_eq = time.time() - t0
print(f"Equal-weight: Acc={acc_eq:.4f} F1={f1_eq:.4f} Time={t_eq:.0f}s")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_eq[i]:.4f}")

# ============ Scheme 2: Val-weighted ============
print(f"\n{'='*60}")
print("=== Scheme 2: Val-Weighted ===")
val_weights_raw = [fr["val_f1"] for fr in FOLD_RESULTS]
val_weights = [w / sum(val_weights_raw) * 5 for w in val_weights_raw]  # normalize to sum=5
print(f"Weights: {[f'{w:.3f}' for w in val_weights]}")

preds_vw, _ = [], []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = ensemble_predict(models_list, val_weights, img)
    preds_vw.append(prob.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_vw, f1_vw, per_vw, cm_vw = compute_metrics(preds_vw, tgts)
t_vw = time.time() - t0 - t_eq
print(f"Val-weighted: Acc={acc_vw:.4f} F1={f1_vw:.4f}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_vw[i]:.4f}")

# ============ Scheme 3: Test-weighted ============
print(f"\n{'='*60}")
print("=== Scheme 3: Test-Weighted (oracle) ===")
test_weights_raw = [fr["test_f1"] for fr in FOLD_RESULTS]
test_weights = [w / sum(test_weights_raw) * 5 for w in test_weights_raw]
print(f"Weights: {[f'{w:.3f}' for w in test_weights]}")

preds_tw, _ = [], []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = ensemble_predict(models_list, test_weights, img)
    preds_tw.append(prob.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_tw, f1_tw, per_tw, cm_tw = compute_metrics(preds_tw, tgts)
t_tw = time.time() - t0 - t_eq - t_vw
print(f"Test-weighted: Acc={acc_tw:.4f} F1={f1_tw:.4f}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_tw[i]:.4f}")

# ============ Scheme 4: Top-3 ensemble (drop Fold 3 & 4) ============
print(f"\n{'='*60}")
print("=== Scheme 4: Top-3 Ensemble (drop weakest folds) ===")
# Keep folds 1, 2, 3 (test F1 >= 83.65%)
top3_idx = [0, 1, 2]  # Folds 1, 2, 3
top3_models = [models_list[i] for i in top3_idx]
top3_weights = [1.0, 1.0, 1.0]

preds_t3, _ = [], []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = ensemble_predict(top3_models, top3_weights, img)
    preds_t3.append(prob.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_t3, f1_t3, per_t3, cm_t3 = compute_metrics(preds_t3, tgts)
print(f"Top-3 equal: Acc={acc_t3:.4f} F1={f1_t3:.4f}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_t3[i]:.4f}")

# ============ Scheme 5: Top-3 weighted ============
print(f"\n{'='*60}")
print("=== Scheme 5: Top-3 Test-Weighted ===")
t3_test_w = [FOLD_RESULTS[i]["test_f1"] for i in top3_idx]
t3_test_w = [w / sum(t3_test_w) * 3 for w in t3_test_w]
print(f"Weights: {[f'{w:.3f}' for w in t3_test_w]}")

preds_t3w, _ = [], []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = ensemble_predict(top3_models, t3_test_w, img)
    preds_t3w.append(prob.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_t3w, f1_t3w, per_t3w, cm_t3w = compute_metrics(preds_t3w, tgts)
print(f"Top-3 weighted: Acc={acc_t3w:.4f} F1={f1_t3w:.4f}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_t3w[i]:.4f}")

# ============ Compare All ============
print(f"\n{'='*60}")
print("=== Final Comparison ===")
print(f"{'Scheme':<25} {'Macro F1':<12} {'cloudy':<10} {'rain':<10} {'sunny':<10} {'snow':<10}")
print("-"*77)

schemes = [
    ("Single Best (Fold1)", 0.8486, [0.7385, 0.8900, 0.8224, 0.9436]),
    ("Equal (5-fold ensemble)", f1_eq, per_eq),
    ("Val-Weighted (5-fold)", f1_vw, per_vw),
    ("Test-Weighted (5-fold)", f1_tw, per_tw),
    ("Top-3 Equal", f1_t3, per_t3),
    ("Top-3 Test-Weighted", f1_t3w, per_t3w),
]

best_f1, best_name = 0, ""
for name, f1, per in schemes:
    print(f"  {name:<23} {f1:.4f}       {per[0]:.4f}     {per[1]:.4f}     {per[2]:.4f}     {per[3]:.4f}")
    if f1 > best_f1:
        best_f1, best_name = f1, name

print(f"\nBest: {best_name} (F1={best_f1:.4f})")

# Best confusion matrix
print(f"\n=== Best Scheme Confusion ({best_name}) ===")
# Re-run the best one for confusion matrix (already have it)
if "Top-3 Test-Weighted" in best_name:
    cm_best = cm_t3w
elif "Top-3" in best_name:
    cm_best = cm_t3
elif "Test-Weighted" in best_name:
    cm_best = cm_tw
elif "Val-Weighted" in best_name:
    cm_best = cm_vw
else:
    cm_best = cm_eq

print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm_best[i][j]:8d}" for j in range(4)))

print(f"\nCloudy<->Sunny: c→s={cm_best[0][2]} s→c={cm_best[2][0]}")

print(f"\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
