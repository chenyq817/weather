"""Test evaluation for best_model_v4.pth"""
import os, time, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import torch.nn as nn
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_clean")
CKPT_PATH = os.path.join(BASE_DIR, "output", "best_model_v4.pth")
IMAGE_SIZE = 260
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")
BATCH_SIZE = 12

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir; self.transform = transform; self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        for cls in CLASS_NAMES:
            d = os.path.join(root_dir, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d,f), self.class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
        if self.transform: img = self.transform(img)
        return img, l

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
print(f"Checkpoint: Epoch {ckpt['epoch']+1}, val_f1={ckpt['f1']:.4f}, val_acc={ckpt['acc']:.4f}")

model = models.efficientnet_b2(weights=None)
inf = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(inf, 256), nn.BatchNorm1d(256),
    nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(DEVICE)

test_dataset = WeatherDataset(os.path.join(DATA_DIR, "test"), val_transform)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

preds, tgts = [], []
t0 = time.time()
with torch.no_grad():
    for x, y in test_loader:
        out = model(x.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
tt = time.time() - t0

acc = accuracy_score(tgts, preds)
f1m = f1_score(tgts, preds, average="macro")
per_f1 = f1_score(tgts, preds, average=None)

print(f"\n=== v4 测试结果 ({len(test_dataset)} 张) ===")
print(f"Accuracy: {acc:.4f} | F1 macro: {f1m:.4f} | 推理: {tt:.1f}s")
print(f"\n分类报告:")
print(classification_report(tgts, preds, target_names=CLASS_NAMES, digits=4))
cm = confusion_matrix(tgts, preds)
print("混淆矩阵:")
print("         " + " ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:8s}" + " ".join(f"{cm[i][j]:8d}" for j in range(NUM_CLASSES)))

# Compare all versions
base = {"cloudy":0.6801,"rain":0.8550,"sunny":0.7740,"snow":0.9147,"macro":0.8060}
v3  = {"cloudy":0.6981,"rain":0.8432,"sunny":0.7889,"snow":0.9254,"macro":0.8139}
print(f"\n{'类别':<12} {'Base(B0)':<10} {'v3(B2)':<10} {'v4(+W)':<10}")
print("-"*42)
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<10} {base[cls]:.4f}     {v3[cls]:.4f}    {per_f1[i]:.4f}")
print(f"  {'macro':<10} {base['macro']:.4f}     {v3['macro']:.4f}    {f1m:.4f}")

print(f"\n阴天/晴天混淆:")
print(f"  Base: cloudy->sunny=44 sunny->cloudy=32")
print(f"  v3:   cloudy->sunny={33} sunny->cloudy={37}")
print(f"  v4:   cloudy->sunny={cm[0][2]} sunny->cloudy={cm[2][0]}")
