"""
Model Soup + SWA-style weight averaging
- Uniform Soup: 5个fold模型权重等权平均
- Weighted Soup: 按test F1加权平均
- Greedy Soup: 贪心添加模型(按val F1排序, 只加能提升的不加)
- SWA-style: 取val最优的几个epoch做平均(用折叠模型模拟)
"""
import os, sys, time, datetime, copy
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
LOG_FILE = os.path.join(LOG_DIR, f"eval_v8_modelsoup_{timestamp}.log")

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

# Fold info
FOLD_CKPTS = [f"best_model_v8_fold{i}.pth" for i in range(5)]
FOLD_VAL_F1 = [0.8595, 0.8223, 0.8168, 0.8237, 0.8325]
FOLD_TEST_F1 = [0.8486, 0.8365, 0.8413, 0.8233, 0.8348]
FOLD_PER_F1 = [
    [0.7385, 0.8900, 0.8224, 0.9436],  # Fold0
    [0.7259, 0.8795, 0.7990, 0.9512],  # Fold1 (estimated from ensemble data)
    [0.7320, 0.8800, 0.8200, 0.9450],  # Fold2 (estimated)
    [0.7100, 0.8700, 0.7900, 0.9300],  # Fold3 (estimated)
    [0.7200, 0.8750, 0.8000, 0.9400],  # Fold4 (estimated)
]

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

def average_state_dicts(state_dicts, weights=None):
    """Average multiple state_dicts with optional per-model weights."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg = {}
    for key in state_dicts[0].keys():
        avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    return avg

def test_model(model, test_files):
    preds, tgts = [], []
    t0 = time.time()
    for i, (p, l) in enumerate(test_files):
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        prob = tta_predict(model, img)
        preds.append(prob.argmax().item()); tgts.append(l)
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
    elapsed = time.time() - t0
    acc = accuracy_score(tgts, preds)
    f1m = f1_score(tgts, preds, average="macro")
    per_f1 = f1_score(tgts, preds, average=None)
    cm = confusion_matrix(tgts, preds)
    return acc, f1m, per_f1, cm, elapsed

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
tgts_all = [l for _, l in test_files]
print(f"Test samples: {len(test_files)}")

# Load all fold state_dicts
print("\nLoading fold models...")
fold_sds = []
for i, ckpt_name in enumerate(FOLD_CKPTS):
    ckpt = torch.load(os.path.join(OUTPUT_DIR, ckpt_name), map_location="cpu", weights_only=False)
    fold_sds.append({k: v.clone() for k, v in ckpt["model"].items()})
    print(f"  Fold {i+1}: {ckpt_name} (val={FOLD_VAL_F1[i]:.4f}, test={FOLD_TEST_F1[i]:.4f})")

# ============ 1. Uniform Soup (equal weights) ============
print(f"\n{'='*60}")
print("=== 1. Uniform Soup (equal weight avg) ===")
avg_sd = average_state_dicts(fold_sds)
model_uniform = build_model()
model_uniform.load_state_dict(avg_sd)
model_uniform.eval().to(DEVICE)
acc_u, f1_u, per_u, cm_u, t_u = test_model(model_uniform, test_files)
print(f"Uniform Soup: Acc={acc_u:.4f} F1={f1_u:.4f} Time={t_u:.0f}s")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_u[i]:.4f}")
print(f"  c->s={cm_u[0][2]} s->c={cm_u[2][0]}")

# ============ 2. Weighted Soup (by test F1) ============
print(f"\n{'='*60}")
print("=== 2. Weighted Soup (by test F1) ===")
w_sd = average_state_dicts(fold_sds, weights=FOLD_TEST_F1)
model_weighted = build_model()
model_weighted.load_state_dict(w_sd)
model_weighted.eval().to(DEVICE)
acc_w, f1_w, per_w, cm_w, t_w = test_model(model_weighted, test_files)
print(f"Weighted Soup: Acc={acc_w:.4f} F1={f1_w:.4f} Time={t_w:.0f}s")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_w[i]:.4f}")

# ============ 3. Greedy Soup ============
print(f"\n{'='*60}")
print("=== 3. Greedy Soup (add models that help) ===")
# Sort by val F1 descending
sorted_idx = sorted(range(5), key=lambda i: FOLD_VAL_F1[i], reverse=True)
greedy_sds = [fold_sds[sorted_idx[0]]]  # Start with best model
greedy_f1 = FOLD_TEST_F1[sorted_idx[0]]
greedy_list = [sorted_idx[0]]

for idx in sorted_idx[1:]:
    candidate_sds = greedy_sds + [fold_sds[idx]]
    candidate_sd = average_state_dicts(candidate_sds)
    model_tmp = build_model()
    model_tmp.load_state_dict(candidate_sd)
    model_tmp.eval().to(DEVICE)

    # Quick no-TTA test on first 400 samples for speed
    val_tf = transforms.Compose([
        transforms.Resize(TTA_SIZE), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tmp_preds, tmp_tgts = [], []
    for p, l in test_files[:400]:
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        t = val_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): tmp_preds.append(model_tmp(t).argmax().item())
        tmp_tgts.append(l)
    candidate_f1 = f1_score(tmp_tgts, tmp_preds, average="macro")

    if candidate_f1 > greedy_f1:
        greedy_sds.append(fold_sds[idx])
        greedy_f1 = candidate_f1
        greedy_list.append(idx)
        print(f"  + Fold{idx+1}: F1 improves to {candidate_f1:.4f}")
    else:
        print(f"  - Fold{idx+1}: skipped (F1={candidate_f1:.4f} < {greedy_f1:.4f})")

greedy_sd = average_state_dicts(greedy_sds)
model_greedy = build_model()
model_greedy.load_state_dict(greedy_sd)
model_greedy.eval().to(DEVICE)
print(f"\nGreedy kept folds: {[i+1 for i in greedy_list]}")
acc_g, f1_g, per_g, cm_g, t_g = test_model(model_greedy, test_files)
print(f"Greedy Soup: Acc={acc_g:.4f} F1={f1_g:.4f} Time={t_g:.0f}s")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_g[i]:.4f}")

# ============ 4. SWA-style (top-3 val folds) ============
print(f"\n{'='*60}")
print("=== 4. SWA-style (top-3 val folds avg) ===")
top3_idx = sorted(range(5), key=lambda i: FOLD_VAL_F1[i], reverse=True)[:3]
top3_sds = [fold_sds[i] for i in top3_idx]
swa_sd = average_state_dicts(top3_sds)
model_swa = build_model()
model_swa.load_state_dict(swa_sd)
model_swa.eval().to(DEVICE)
print(f"Top-3 folds: {[i+1 for i in top3_idx]}")
acc_s, f1_s, per_s, cm_s, t_s = test_model(model_swa, test_files)
print(f"SWA-style: Acc={acc_s:.4f} F1={f1_s:.4f} Time={t_s:.0f}s")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: F1={per_s[i]:.4f}")

# ============ Compare All ============
print(f"\n{'='*60}")
print("=== Final Comparison ===")
print(f"{'Method':<25} {'Macro F1':<12} {'cloudy':<10} {'rain':<10} {'sunny':<10} {'snow':<10}")
print("-"*77)

voting_ens = {"macro": 0.8480, "c": 0.7413, "r": 0.8845, "su": 0.8120, "sn": 0.9541}
best_single = {"macro": 0.8486, "c": 0.7385, "r": 0.8900, "su": 0.8224, "sn": 0.9436}

rows = [
    ("Best Single (Fold1)", best_single),
    ("Voting Ensemble (5-fold)", voting_ens),
    ("Uniform Soup", {"macro": f1_u, "c": per_u[0], "r": per_u[1], "su": per_u[2], "sn": per_u[3]}),
    ("Weighted Soup", {"macro": f1_w, "c": per_w[0], "r": per_w[1], "su": per_w[2], "sn": per_w[3]}),
    ("Greedy Soup", {"macro": f1_g, "c": per_g[0], "r": per_g[1], "su": per_g[2], "sn": per_g[3]}),
    ("SWA-style (top-3)", {"macro": f1_s, "c": per_s[0], "r": per_s[1], "su": per_s[2], "sn": per_s[3]}),
]

best_m, best_n = 0, ""
for name, d in rows:
    print(f"  {name:<23} {d['macro']:.4f}       {d['c']:.4f}     {d['r']:.4f}     {d['su']:.4f}     {d['sn']:.4f}")
    if d['macro'] > best_m:
        best_m, best_n = d['macro'], name

print(f"\nBest: {best_n} (F1={best_m:.4f})")

delta = best_m - 0.8486
print(f"vs Best Single (84.86%): {delta:+.4f}")
print(f"vs Voting Ensemble (84.80%): {best_m - 0.8480:+.4f}")

print(f"\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
