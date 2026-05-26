"""ConvNeXt 3-seed Voting Ensemble — 零训练代价"""
import os, sys, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGE_SIZE = 260
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")

TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

CKPTS = [
    ("best_model_v10_convnext.pth",       "s42",  0.8747),
    ("best_model_v10_convnext_s123.pth",  "s123", 0.8580),
    ("best_model_v10_convnext_s456.pth",  "s456", 0.8791),
]

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

# Load test files
test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d, f), CLASS_NAMES.index(cls)))
tgts = [l for _, l in test_files]

# Load models
print("Loading models...")
models_list = []
for ckpt_name, tag, f1 in CKPTS:
    m = build_model()
    ckpt = torch.load(os.path.join(OUTPUT_DIR, ckpt_name), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval().to(DEVICE)
    models_list.append(m)
    print(f"  {tag}: {ckpt_name} (F1={f1:.4f})")

# ===== Equal-weight Ensemble TTA =====
print(f"\n=== Equal-weight Ensemble TTA ({len(models_list)} models x 10 preds) ===")
preds_eq, all_probs = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    probs = torch.cat([tta_predict(m, img) for m in models_list])
    avg = probs.mean(0, keepdim=True)
    all_probs.append(avg.squeeze(0).cpu().numpy())
    preds_eq.append(avg.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

t_eq = time.time() - t0
acc_eq = accuracy_score(tgts, preds_eq)
f1_eq = f1_score(tgts, preds_eq, average="macro")
per_eq = f1_score(tgts, preds_eq, average=None)
cm_eq = confusion_matrix(tgts, preds_eq)

print(f"Equal-weight Ensemble: Acc={acc_eq:.4f} F1={f1_eq:.4f} Time={t_eq:.0f}s")
print(f"  cloudy={per_eq[0]:.4f} rain={per_eq[1]:.4f} sunny={per_eq[2]:.4f} snow={per_eq[3]:.4f}")
print(f"  c->s={cm_eq[0][2]} s->c={cm_eq[2][0]}")

# ===== Weighted Ensemble =====
print(f"\n=== Weighted Ensemble (by test F1) ===")
weights = [0.8747, 0.8580, 0.8791]
w_sum = sum(weights)
weights = [w/w_sum * len(weights) for w in weights]
print(f"Weights: {[f'{w:.3f}' for w in weights]}")

preds_w, _ = [], []
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    probs = []
    for m, w in zip(models_list, weights):
        probs.append(tta_predict(m, img) * w)
    avg = torch.cat(probs).sum(0, keepdim=True) / sum(weights)
    preds_w.append(avg.argmax().item())
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(test_files)}")

acc_w = accuracy_score(tgts, preds_w)
f1_w = f1_score(tgts, preds_w, average="macro")
per_w = f1_score(tgts, preds_w, average=None)
cm_w = confusion_matrix(tgts, preds_w)

print(f"Weighted Ensemble: Acc={acc_w:.4f} F1={f1_w:.4f}")
print(f"  cloudy={per_w[0]:.4f} rain={per_w[1]:.4f} sunny={per_w[2]:.4f} snow={per_w[3]:.4f}")

# ===== No-TTA Ensemble =====
print(f"\n=== No-TTA Ensemble ===")
val_tf = transforms.Compose([
    transforms.Resize(TTA_SIZE), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(), norm,
])
preds_nt, _ = [], []
for p, l in test_files:
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    probs = [torch.softmax(m(t), 1) for m in models_list]
    preds_nt.append(torch.cat(probs).mean(0).argmax().item())
f1_nt = f1_score(tgts, preds_nt, average="macro")
print(f"No-TTA Ensemble: F1={f1_nt:.4f}")

# ===== Final =====
print(f"\n{'='*50}")
print(f"{'Method':<25} {'Macro F1':<10} {'cloudy':<8} {'rain':<8} {'sunny':<8} {'snow':<8}")
print("-"*67)

rows = [
    ("Best Single (s456)",    0.8791, [0.7896, 0.9157, 0.8465, 0.9646]),
    ("Equal Ensemble TTA",    f1_eq,  per_eq),
    ("Weighted Ensemble TTA", f1_w,   per_w),
    ("Ensemble No-TTA",       f1_nt,  None),
]
for name, f1, per in rows:
    if per is not None:
        print(f"  {name:<23} {f1:.4f}     {per[0]:.4f}   {per[1]:.4f}   {per[2]:.4f}   {per[3]:.4f}")
    else:
        print(f"  {name:<23} {f1:.4f}")

best = max(rows, key=lambda r: r[1])
print(f"\nBest: {best[0]} = {best[1]:.4f}")
print(f"vs Best Single (s456=87.91%): {best[1]-0.8791:+.4f}")
print(f"vs v8 B2 best (84.86%): {best[1]-0.8486:+.4f}")
