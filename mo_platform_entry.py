"""
智海Mo平台训练入口 - ConvNeXt-Tiny + v8增强策略
适配平台标准 train(train_dataset, val_dataset) / predict(test_dataset) 接口
"""
import os, sys, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import f1_score
from PIL import Image

# ============ Config ============
IMAGE_SIZE = 260
BATCH_SIZE = 32
NUM_EPOCHS = 40
WARMUP_EPOCHS = 3
LR = 8e-4; FINE_LR = 3e-5; WD = 3e-4
LABEL_SMOOTH = 0.1; MIXUP_A = 0.3; CUTMIX_A = 0.2; MIXUP_P = 0.7
GRAD_CLIP = 1.0
NUM_CLASSES = 4
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomEqualize(p=0.1), transforms.RandomGrayscale(p=0.05),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============ MixUp / CutMix ============
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, _, H, W = x.size(); idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw = int(W * np.sqrt(1-lam)); bh = int(H * np.sqrt(1-lam))
    x0, y0 = np.clip(cx-bw//2, 0, W), np.clip(cy-bh//2, 0, H)
    x1, y1 = np.clip(cx+bw//2, 0, W), np.clip(cy+bh//2, 0, H)
    mx = x.clone(); mx[:,:,y0:y1,x0:x1] = x[idx,:,y0:y1,x0:x1]
    return mx, y, y[idx], 1 - (x1-x0)*(y1-y0)/(H*W)

# ============ Model ============
def build_model(pretrained_path=None):
    """构建模型，可选加载天气预训练权重"""
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    inf = model.classifier[2].in_features  # 768
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
        nn.Dropout(0.4), nn.Linear(inf, 256),
        nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        # 重新随机初始化分类头 (适应新数据分布)
        model.classifier = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
            nn.Dropout(0.4), nn.Linear(inf, 256),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
        )
        print("Classifier head re-initialized, backbone kept.")
    return model.to(DEVICE)

# ============ Dataset wrapper ============
class PlatformDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, str): img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray): img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# ============ Train (默认: ImageNet从头训练, 同数据集不用预训练) ============
def train(train_dataset, val_dataset=None):
    """默认从ImageNet从头训练 (同数据集不用预训练权重)"""
    return train_from_scratch(train_dataset, val_dataset)

def train_from_scratch(train_dataset, val_dataset=None):
    """从ImageNet从头训练 (不使用天气预训练)"""
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)

    model = build_model()  # ImageNet init
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler() if USE_AMP else None

    for n, p in model.named_parameters(): p.requires_grad = "classifier" in n
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_f1 = 0.0
    for ep in range(NUM_EPOCHS):
        if ep == WARMUP_EPOCHS:
            print(f"E{ep+1}: Unfreeze backbone")
            for p in model.parameters(): p.requires_grad = True
            hp = [p for n,p in model.named_parameters() if "classifier" in n]
            bp = [p for n,p in model.named_parameters() if "classifier" not in n]
            opt = optim.AdamW([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}], weight_decay=WD)
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

        model.train(); tl_ = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            ma = ep >= WARMUP_EPOCHS and torch.rand(1).item() < MIXUP_P
            if ma:
                if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup_data(x, y, MIXUP_A)
                else: x, ya, yb, lam = cutmix_data(x, y, CUTMIX_A)
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
                scaler.scale(loss).backward()
                if GRAD_CLIP>0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            tl_ += loss.item()
        sch.step()

        if val_loader:
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x.to(DEVICE)); vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.numpy())
            f1 = f1_score(vt, vp, average="macro")
            print(f"E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(train_loader):.4f} | f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1; torch.save({"model":model.state_dict(),"f1":f1}, "best_model.pth")
        else:
            print(f"E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(train_loader):.4f}")
    return model

# ============ Train with Pretrained Weights ============
def train_with_pretrained(train_dataset, val_dataset=None,
                           pretrained_path="output/best_model_v10_convnext_s456.pth"):
    """加载天气预训练权重, 在新数据集上微调"""
    PT_LR = 3e-4      # 更低的head LR (backbone已适应天气)
    PT_FINE_LR = 1e-5 # 更低的backbone LR

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)

    print(f"Fine-tuning from: {pretrained_path}")
    model = build_model(pretrained_path=pretrained_path)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler() if USE_AMP else None

    # 直接用分组学习率 (backbone已适应, 不需要warmup冻结)
    for p in model.parameters(): p.requires_grad = True
    hp = [p for n,p in model.named_parameters() if "classifier" in n]
    bp = [p for n,p in model.named_parameters() if "classifier" not in n]
    opt = optim.AdamW([{"params":hp,"lr":PT_LR},{"params":bp,"lr":PT_FINE_LR}], weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)

    best_f1 = 0.0
    for ep in range(NUM_EPOCHS):
        model.train(); tl_ = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            ma = torch.rand(1).item() < MIXUP_P
            if ma:
                if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup_data(x, y, MIXUP_A)
                else: x, ya, yb, lam = cutmix_data(x, y, CUTMIX_A)
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss = lam*criterion(out,ya)+(1-lam)*criterion(out,yb) if ma else criterion(out,y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            tl_ += loss.item()
        sch.step()

        if val_loader:
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x.to(DEVICE)); vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.numpy())
            f1 = f1_score(vt, vp, average="macro")
            print(f"FT E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(train_loader):.4f} | f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1; torch.save({"model":model.state_dict(),"f1":f1}, "best_model.pth")
        else:
            print(f"FT E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(train_loader):.4f}")
    return model

# ============ Predict ============
TTA_SIZE = int(IMAGE_SIZE * 1.14)

def predict(test_dataset, model_path="best_model.pth"):
    model = build_model()
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))

    # TTA transforms
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def tta_predict(img):
        if isinstance(img, np.ndarray): img = Image.fromarray(img)
        elif isinstance(img, str): img = Image.open(img).convert("RGB")
        img_r = transforms.Resize(TTA_SIZE)(img)
        crops = transforms.FiveCrop(IMAGE_SIZE)(img_r)
        probs = []
        for crop in crops:
            for flip in [False, True]:
                c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
                t = norm(to_tensor(c)).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): probs.append(torch.softmax(model(t), 1))
        return torch.cat(probs).mean(0)

    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model.eval(); all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            for i in range(len(images)):
                prob = tta_predict(images[i].cpu().numpy() if torch.is_tensor(images[i]) else images[i])
                all_preds.append(prob.argmax().item())
    return all_preds
