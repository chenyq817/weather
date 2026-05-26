"""Apply TTA to best_model_v5_gpu.pth - no training, just inference"""
import os, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE_PATH, "output", "best_model_v5_gpu.pth")
DATA_DIR = os.path.join(BASE_PATH, "data", "weather_clean")
IMAGE_SIZE = 260
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
DEVICE = torch.device("cuda")

TTA_SIZE = int(IMAGE_SIZE * 1.14)
val_tf = transforms.Compose([
    transforms.Resize(TTA_SIZE), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
to_tensor = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

# Load model
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
print(f"v5 checkpoint: Epoch {ckpt['epoch']+1}, val_f1={ckpt['f1']:.4f}")

model = models.efficientnet_b2(weights=None)
inf = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(inf, 256), nn.BatchNorm1d(256),
    nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, len(CLASS_NAMES)),
)
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)

# Test files
test_files = []
for cls in CLASS_NAMES:
    d = os.path.join(DATA_DIR, "test", cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                test_files.append((os.path.join(d,f), CLASS_NAMES.index(cls)))

# TTA
print(f"Running TTA on {len(test_files)} images (10 variants each)...")
preds_tta, tgts = [], []
t0 = time.time()
for i, (p, l) in enumerate(test_files):
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    prob = tta_predict(model, img)
    preds_tta.append(prob.argmax().item()); tgts.append(l)
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(test_files)}")
tt = time.time() - t0

# No-TTA
print("Running No-TTA...")
preds_no = []
for p, l in test_files:
    try: img = Image.open(p).convert("RGB")
    except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): preds_no.append(model(t).argmax().item())

# Results
acc_tta = accuracy_score(tgts, preds_tta); f1_tta = f1_score(tgts, preds_tta, average="macro")
acc_no = accuracy_score(tgts, preds_no); f1_no = f1_score(tgts, preds_no, average="macro")
per_tta = f1_score(tgts, preds_tta, average=None)
per_no = f1_score(tgts, preds_no, average=None)

print(f"\n{'':<12} {'No-TTA':<12} {'TTA':<12} {'Gain':<12}")
print("-"*48)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<10} {per_no[i]:.4f}       {per_tta[i]:.4f}     {per_tta[i]-per_no[i]:+.4f}")
print(f"  {'macro':<10} {f1_no:.4f}       {f1_tta:.4f}     {f1_tta-f1_no:+.4f}")
print(f"  {'acc':<10} {acc_no:.4f}       {acc_tta:.4f}     {acc_tta-acc_no:+.4f}")

print(f"\nTTA Time: {tt:.1f}s ({tt/len(test_files)*1000:.1f}ms/img)")

cm = confusion_matrix(tgts, preds_tta)
print(f"\nTTA Confusion:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(4)))

# Final comparison
print(f"\n=== Final Leaderboard ===")
base = {"cloudy":0.6801,"rain":0.8550,"sunny":0.7740,"snow":0.9147,"macro":0.8060}
v5  = {"cloudy":0.7053,"rain":0.8676,"sunny":0.8104,"snow":0.9436,"macro":0.8317}
print(f"{'':<12} {'Base(B0)':<10} {'v5(GPU)':<10} {'v5+TTA':<10}")
print("-"*42)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<10} {base[cls]:.4f}     {v5[cls]:.4f}    {per_tta[i]:.4f}")
print(f"  {'macro':<10} {base['macro']:.4f}     {v5['macro']:.4f}    {f1_tta:.4f}")

print(f"\nCloudy->Sunny: Base=44 v5=41 v5+TTA={cm[0][2]}")
print(f"Sunny->Cloudy: Base=32 v5=25 v5+TTA={cm[2][0]}")
