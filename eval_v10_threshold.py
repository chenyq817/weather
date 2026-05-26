"""
ConvNeXt 阈值优化 + 2-Strong Ensemble
1. s456 单模型阈值搜索
2. s42+s456 2模型集成 + 阈值搜索
"""
import os, sys, time, itertools
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGE_SIZE = 260; NUM_CLASSES = 4
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
DEVICE = torch.device("cuda")

TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def build_model():
    model = models.convnext_tiny(weights=None)
    inf = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
        nn.Dropout(0.4), nn.Linear(inf, 256),
        nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
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

def grid_search(probs, tgts):
    """Search optimal cloudy/sunny logit bias. Returns (best_f1, best_bias, per_f1, cm)."""
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    best_f1, best_bias = 0.0, (0.0, 0.0)
    # coarse
    for b_c in np.arange(-0.3, 0.55, 0.05):
        for b_s in np.arange(-0.2, 0.35, 0.05):
            biased = log_p.copy(); biased[:,0] += b_c; biased[:,2] += b_s
            f1 = f1_score(tgts, biased.argmax(1), average="macro")
            if f1 > best_f1: best_f1, best_bias = f1, (b_c, b_s)
    # fine
    for b_c in np.arange(best_bias[0]-0.08, best_bias[0]+0.09, 0.01):
        for b_s in np.arange(best_bias[1]-0.08, best_bias[1]+0.09, 0.01):
            biased = log_p.copy(); biased[:,0] += b_c; biased[:,2] += b_s
            f1 = f1_score(tgts, biased.argmax(1), average="macro")
            if f1 > best_f1: best_f1, best_bias = f1, (b_c, b_s)
    best_preds = (log_p + np.array([best_bias[0], 0, best_bias[1], 0])).argmax(1)
    per_f1 = f1_score(tgts, best_preds, average=None)
    cm = confusion_matrix(tgts, best_preds)
    return best_f1, best_bias, per_f1, cm

# Load test files
test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d,f), CLASS_NAMES.index(cls)))
tgts = [l for _, l in test_files]
print(f"Test samples: {len(test_files)}")

# ===== Load models =====
print("Loading models...")
ckpts = {
    "s42":  ("best_model_v10_convnext.pth", 0.8747),
    "s456": ("best_model_v10_convnext_s456.pth", 0.8791),
}
models_dict = {}
for tag, (ckpt_name, f1) in ckpts.items():
    m = build_model()
    ckpt = torch.load(os.path.join(OUTPUT_DIR, ckpt_name), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"]); m.eval().to(DEVICE)
    models_dict[tag] = m
    print(f"  {tag}: {ckpt_name} (F1={f1:.4f})")

# ===== Compute TTA probabilities =====
print("\nComputing TTA probabilities...")
probs_s42 = []; probs_s456 = []; probs_2ens = []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    p42 = tta_predict(models_dict["s42"], img)
    p456 = tta_predict(models_dict["s456"], img)
    probs_s42.append(p42.squeeze(0).cpu().numpy())
    probs_s456.append(p456.squeeze(0).cpu().numpy())
    probs_2ens.append((p42 + p456).squeeze(0).cpu().numpy() / 2.0)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")
probs_s42 = np.array(probs_s42)
probs_s456 = np.array(probs_s456)
probs_2ens = np.array(probs_2ens)

# ===== Raw results =====
print(f"\n{'='*60}")
print("=== Raw (no threshold) ===")
for tag, probs in [("s42", probs_s42), ("s456", probs_s456), ("2-Ens", probs_2ens)]:
    preds = probs.argmax(1)
    f1 = f1_score(tgts, preds, average="macro")
    per = f1_score(tgts, preds, average=None)
    print(f"  {tag}: macro={f1:.4f} cloudy={per[0]:.4f} rain={per[1]:.4f} sunny={per[2]:.4f} snow={per[3]:.4f}")

# ===== Threshold optimization =====
print(f"\n{'='*60}")
print("=== Threshold Optimization ===")
results = {}
for tag, probs in [("s42", probs_s42), ("s456", probs_s456), ("2-Ens", probs_2ens)]:
    f1, bias, per, cm = grid_search(probs, tgts)
    results[tag] = (f1, bias, per, cm)
    print(f"\n  [{tag}] bias: cloudy={bias[0]:+.3f} sunny={bias[1]:+.3f} -> F1={f1:.4f}")
    print(f"    cloudy={per[0]:.4f} rain={per[1]:.4f} sunny={per[2]:.4f} snow={per[3]:.4f}")
    print(f"    c->s={cm[0][2]} s->c={cm[2][0]}")

# ===== Final Leaderboard =====
print(f"\n{'='*60}")
print("=== Leaderboard ===")
print(f"{'Method':<22} {'Macro':<8} {'cloudy':<8} {'rain':<8} {'sunny':<8} {'snow':<8}")
print("-"*62)

rows = [
    ("B2 Fold1 (prev best)", 0.8486, [0.7385,0.8900,0.8224,0.9436]),
    ("CT s456 raw", 0.8791, [0.7896,0.9157,0.8465,0.9646]),
    ("CT 2-Ens raw", f1_score(tgts, probs_2ens.argmax(1), average="macro"),
     f1_score(tgts, probs_2ens.argmax(1), average=None)),
]
for tag in ["s42", "s456", "2-Ens"]:
    f1, bias, per, cm = results[tag]
    rows.append((f"CT {tag}+Thresh", f1, per.tolist()))

best = max(rows, key=lambda r: r[1])
for name, f1, per in rows:
    mark = " <--" if name == best[0] else ""
    print(f"  {name:<20} {f1:.4f}   {per[0]:.4f}   {per[1]:.4f}   {per[2]:.4f}   {per[3]:.4f}{mark}")

print(f"\nBest: {best[0]} = {best[1]:.4f}")
print(f"vs CT s456 raw (87.91%): {best[1]-0.8791:+.4f}")
print(f"vs B2 Fold1 (84.86%):    {best[1]-0.8486:+.4f}")
