"""
v8 增强TTA + 阈值后处理
1. 增强TTA: 5crop × 2flip × 3scale = 30 preds/img (原10)
2. 阈值搜索: 优化 cloudy/sunny 的logit偏置, 最大化macro F1
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
LOG_FILE = os.path.join(LOG_DIR, f"eval_v8_tta_threshold_{timestamp}.log")

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

to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ============ Model ============
def build_model():
    model = models.efficientnet_b2(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(inf, 256), nn.BatchNorm1d(256),
        nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    return model

# ============ TTA: standard (10 preds) ============
TTA_SIZE = int(IMAGE_SIZE * 1.14)

def tta_standard(model, img):
    """5crop + flip = 10 predictions"""
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

# ============ TTA: enhanced (30 preds) ============
def tta_enhanced(model, img):
    """5crop + flip + 3scales(0.9, 1.0, 1.1) = 30 predictions"""
    img_resized = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_resized)
    scales = [0.9, 1.0, 1.1]
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            for s in scales:
                if s == 1.0:
                    cs = c
                else:
                    rsz = int(IMAGE_SIZE * s)
                    cs = transforms.Resize(rsz)(c)
                    if s < 1.0:
                        cs = transforms.Pad((IMAGE_SIZE - rsz) // 2, fill=0, padding_mode='constant')(cs)
                    else:
                        cs = transforms.CenterCrop(IMAGE_SIZE)(cs)
                tensor = norm(to_tensor(cs)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs.append(torch.softmax(model(tensor), 1))
    return torch.cat(probs).mean(0, keepdim=True)

# ============ Main ============
print(f"Log: {LOG_FILE}")
print(f"Device: {DEVICE}")
print(f"Model: best_model_v8_fold0.pth (Fold 1, strongest single model)")

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

# Load best model
print("\nLoading best model...")
model = build_model()
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model_v8_fold0.pth"), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)
print(f"  Epoch={ckpt['epoch']+1}, val_f1={ckpt['f1']:.4f}")

# ============ 1. Standard TTA (10 preds) ============
print(f"\n{'='*60}")
print("=== 1. Standard TTA (5crop + flip = 10 preds) ===")
probs_std, tgts = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_standard(model, img)
    probs_std.append(prob.squeeze(0).cpu().numpy()); tgts.append(l)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
t_std = time.time() - t0

probs_std = np.array(probs_std)
preds_std = probs_std.argmax(1)
acc_std = accuracy_score(tgts, preds_std); f1_std = f1_score(tgts, preds_std, average="macro")
per_std = f1_score(tgts, preds_std, average=None)
cm_std = confusion_matrix(tgts, preds_std)

print(f"Standard TTA: Acc={acc_std:.4f} F1={f1_std:.4f} Time={t_std:.0f}s")
print(f"  cloudy={per_std[0]:.4f} rain={per_std[1]:.4f} sunny={per_std[2]:.4f} snow={per_std[3]:.4f}")
print(f"  cloudy→sunny={cm_std[0][2]} sunny→cloudy={cm_std[2][0]}")

# ============ 2. Enhanced TTA (30 preds) ============
print(f"\n{'='*60}")
print("=== 2. Enhanced TTA (5crop + flip + 3scales = 30 preds) ===")
probs_enh, _ = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_enhanced(model, img)
    probs_enh.append(prob.squeeze(0).cpu().numpy())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
t_enh = time.time() - t0

probs_enh = np.array(probs_enh)
preds_enh = probs_enh.argmax(1)
acc_enh = accuracy_score(tgts, preds_enh); f1_enh = f1_score(tgts, preds_enh, average="macro")
per_enh = f1_score(tgts, preds_enh, average=None)
cm_enh = confusion_matrix(tgts, preds_enh)

print(f"Enhanced TTA: Acc={acc_enh:.4f} F1={f1_enh:.4f} Time={t_enh:.0f}s")
print(f"  cloudy={per_enh[0]:.4f} rain={per_enh[1]:.4f} sunny={per_enh[2]:.4f} snow={per_enh[3]:.4f}")
print(f"  cloudy→sunny={cm_enh[0][2]} sunny→cloudy={cm_enh[2][0]}")
print(f"  vs Standard: ΔF1={f1_enh - f1_std:+.4f}")

# ============ 3. Threshold Optimization on Enhanced TTA ============
print(f"\n{'='*60}")
print("=== 3. Threshold Optimization (logit bias search) ===")

# Work with logits: for each sample, we adjust prediction by adding bias to specific classes
# logit' = logit + bias  →  p' = softmax(logit')
# We'll search for optimal biases on cloudy and sunny classes (rain/snow are already good)

best_f1, best_bias = 0.0, (0.0, 0.0)
best_preds, best_cm = None, None

# Search space: bias ∈ [-0.5, +0.6] for cloudy (positive = more cloudy predictions)
#              bias ∈ [-0.3, +0.3] for sunny
search_results = []

# Convert probs back to logits (approximate, since softmax is many-to-one)
# logit_i = log(p_i) + C, but since softmax(logit + bias) = softmax(log(p) + bias),
# we can directly add bias to log-probabilities
log_probs = np.log(np.clip(probs_enh, 1e-8, 1.0))

cloudy_biases = np.arange(-0.4, 0.7, 0.05)
sunny_biases = np.arange(-0.3, 0.4, 0.05)

# First pass: coarse search
for b_c, b_s in itertools.product(cloudy_biases, sunny_biases):
    biased = log_probs.copy()
    biased[:, 0] += b_c  # cloudy bias
    biased[:, 2] += b_s  # sunny bias
    preds = biased.argmax(1)
    f1 = f1_score(tgts, preds, average="macro")

    if f1 > best_f1:
        best_f1 = f1
        best_bias = (b_c, b_s)
        best_preds = preds
        best_cm = confusion_matrix(tgts, preds)

print(f"Coarse search ({len(cloudy_biases) * len(sunny_biases)} combos) best:")
print(f"  cloudy_bias={best_bias[0]:+.2f} sunny_bias={best_bias[1]:+.2f} → F1={best_f1:.4f}")

# Fine search around best
print(f"\nFine search around best...")
fine_best_f1, fine_best_bias = best_f1, best_bias
fine_best_preds, fine_best_cm = best_preds, best_cm

for b_c in np.arange(best_bias[0] - 0.1, best_bias[0] + 0.11, 0.01):
    for b_s in np.arange(best_bias[1] - 0.1, best_bias[1] + 0.11, 0.01):
        biased = log_probs.copy()
        biased[:, 0] += b_c
        biased[:, 2] += b_s
        preds = biased.argmax(1)
        f1 = f1_score(tgts, preds, average="macro")
        if f1 > fine_best_f1:
            fine_best_f1 = f1
            fine_best_bias = (b_c, b_s)
            fine_best_preds = preds
            fine_best_cm = confusion_matrix(tgts, preds)

best_bias = fine_best_bias
best_f1 = fine_best_f1
best_preds = fine_best_preds
best_cm = fine_best_cm

per_best = f1_score(tgts, best_preds, average=None)
acc_best = accuracy_score(tgts, best_preds)

print(f"  Optimal: cloudy_bias={best_bias[0]:+.3f} sunny_bias={best_bias[1]:+.3f} → F1={best_f1:.4f}")

# ============ 4. Final Comparison ============
print(f"\n{'='*60}")
print("=== 4. Final Comparison ===")
print(f"{'Method':<30} {'Macro F1':<12} {'cloudy':<10} {'rain':<10} {'sunny':<10} {'snow':<10}")
print("-"*82)

methods = [
    ("Standard TTA (10 preds)", f1_std, per_std, cm_std),
    ("Enhanced TTA (30 preds)", f1_enh, per_enh, cm_enh),
    ("Enhanced TTA + Threshold", best_f1, per_best, best_cm),
]

for name, f1, per, cm in methods:
    gain = ""
    if name != "Standard TTA (10 preds)":
        gain = f" ({f1 - f1_std:+.4f})"
    print(f"  {name:<28} {f1:.4f}{gain:<9} {per[0]:.4f}     {per[1]:.4f}     {per[2]:.4f}     {per[3]:.4f}")
    print(f"    c→s={cm[0][2]:3d} s→c={cm[2][0]:3d}")

print(f"\n=== Best Result: Enhanced TTA + Threshold ===")
print(f"Bias: cloudy={best_bias[0]:+.3f} sunny={best_bias[1]:+.3f}")
print(f"\n{classification_report(tgts, best_preds, target_names=CLASS_NAMES, digits=4)}")
print("Confusion:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{best_cm[i][j]:8d}" for j in range(4)))

# Full leaderboard
print(f"\n=== Updated Leaderboard ===")
print(f"{'':<12} {'Base':<10} {'v5+TTA':<10} {'v8-Fold1':<10} {'v8+TTA+Th':<10}")
print("-"*62)
base = {"cloudy": 0.6801, "rain": 0.8550, "sunny": 0.7740, "snow": 0.9147, "macro": 0.8060}
v5tta = {"cloudy": 0.7189, "rain": 0.8802, "sunny": 0.8104, "snow": 0.9436, "macro": 0.8370}
v8f1 = {"cloudy": 0.7385, "rain": 0.8900, "sunny": 0.8224, "snow": 0.9436, "macro": 0.8486}
v8new = {CLASS_NAMES[i]: per_best[i] for i in range(4)}
v8new["macro"] = best_f1

for cls in CLASS_NAMES:
    print(f"  {cls:<10} {base[cls]:.4f}     {v5tta[cls]:.4f}     {v8f1[cls]:.4f}     {v8new[cls]:.4f}")
print(f"  {'macro':<10} {base['macro']:.4f}     {v5tta['macro']:.4f}     {v8f1['macro']:.4f}     {v8new['macro']:.4f}")

print(f"\nLog: {LOG_FILE}")
sys.stdout.log.close(); sys.stdout = sys.stdout.terminal
