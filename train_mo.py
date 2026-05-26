"""
MO平台训练脚本 - 从 -d/weather_classification/ 读取数据, 加载预训练权重微调
原始6类: cloudy, rainy, sunny, snow, haze, thunder → 取4类: cloudy, rainy, sunny, snow
数据量: 可配置, 默认 800 train / 200 test 每类 (小规模测试)
"""
import os, sys, time, datetime, random, shutil
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============ 可配置参数 ============
DATA_DIR = "-d/weather_classification"          # MO平台数据路径
TRAIN_PER_CLASS = 800                           # 训练集每类数量
TEST_PER_CLASS = 200                            # 测试集每类数量
PRETRAINED_MODEL = "output/best_model_v10_convnext_s456.pth"  # 预训练权重(上传到MO平台)
OUTPUT_MODEL = "best_model_mo.pth"              # 输出模型名

IMAGE_SIZE = 260; BATCH_SIZE = 32; NUM_EPOCHS = 30
LR = 3e-4; FINE_LR = 1e-5; WD = 3e-4           # 更低LR (backbone已预训练)
LABEL_SMOOTH = 0.1; MIXUP_A = 0.3; CUTMIX_A = 0.2; MIXUP_P = 0.7
GRAD_CLIP = 1.0

# 类别映射: 原始类名 → 训练类名
CLASS_MAP = {"cloudy": "cloudy", "rainy": "rain", "sunny": "sunny", "snow": "snow"}
TRAIN_CLASSES = ["cloudy", "rain", "sunny", "snow"]
NUM_CLASSES = len(TRAIN_CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Data: {DATA_DIR} | {TRAIN_PER_CLASS}/{TEST_PER_CLASS} per class")
print(f"Pretrained: {PRETRAINED_MODEL}")

# ============ 数据准备 ============
def prepare_data(data_dir, train_per_class, test_per_class):
    """从原始6类数据中提取4类, 划分train/test, 复制到临时目录"""
    print(f"\nPreparing data from {data_dir}...")

    # 收集4类图片
    class_samples = {}
    for src_cls, train_cls in CLASS_MAP.items():
        src_dir = os.path.join(data_dir, src_cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Directory not found: {src_dir}")
        all_files = [f for f in os.listdir(src_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        n = min(train_per_class + test_per_class, len(all_files))
        selected = random.sample(all_files, n)
        class_samples[train_cls] = [(os.path.join(src_dir, f), train_cls) for f in selected]
        print(f"  {src_cls} → {train_cls}: {n} selected from {len(all_files)}")

    # 划分并复制
    tmp_dir = "data_mo_tmp"
    for split, n in [("train", train_per_class), ("test", test_per_class)]:
        for cls in TRAIN_CLASSES:
            os.makedirs(os.path.join(tmp_dir, split, cls), exist_ok=True)

    for cls in TRAIN_CLASSES:
        samples = class_samples[cls]
        random.shuffle(samples)
        train_samples = samples[:train_per_class]
        test_samples = samples[train_per_class:train_per_class + test_per_class]

        for split, split_samples in [("train", train_samples), ("test", test_samples)]:
            dst_dir = os.path.join(tmp_dir, split, cls)
            for src_path, _ in split_samples:
                dst = os.path.join(dst_dir, os.path.basename(src_path))
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)

        print(f"  {cls}: train={len(train_samples)} test={len(test_samples)}")

    return tmp_dir

# ============ 增强 ============
train_tf = transforms.Compose([
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
val_tf = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============ Dataset ============
class WeatherDS(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []; self.tf = transform
        for cls in TRAIN_CLASSES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        self.samples.append((os.path.join(d,f), TRAIN_CLASSES.index(cls)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0))
        if self.tf: img = self.tf(img)
        return img, l

# ============ MixUp / CutMix ============
def mixup(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam
def cutmix(x, y, a):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    B, _, H, W = x.size(); idx = torch.randperm(B, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw = int(W*np.sqrt(1-lam)); bh = int(H*np.sqrt(1-lam))
    x0, y0 = np.clip(cx-bw//2,0,W), np.clip(cy-bh//2,0,H)
    x1, y1 = np.clip(cx+bw//2,0,W), np.clip(cy+bh//2,0,H)
    mx = x.clone(); mx[:,:,y0:y1,x0:x1] = x[idx,:,y0:y1,x0:x1]
    return mx, y, y[idx], 1-(x1-x0)*(y1-y0)/(H*W)

# ============ Model ============
def build_model(pretrained_path=None):
    model = models.convnext_tiny(weights=None)  # no ImageNet, we load our own
    inf = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
        nn.Dropout(0.4), nn.Linear(inf, 256),
        nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        # 重新初始化分类头
        model.classifier = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
            nn.Dropout(0.4), nn.Linear(inf, 256),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
        )
    return model.to(DEVICE)

# ============ TTA ============
TTA_SIZE = int(IMAGE_SIZE * 1.14)
tt_to_tensor = transforms.ToTensor()
tt_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def tta_predict(model, img):
    img_r = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_r)
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            t = tt_norm(tt_to_tensor(c)).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): probs.append(torch.softmax(model(t), 1))
    return torch.cat(probs).mean(0, keepdim=True)

# ============ Main ============
if __name__ == "__main__":
    # Step 1: 准备数据
    data_path = prepare_data(DATA_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS)

    # Step 2: 加载数据
    tr_ds = WeatherDS(os.path.join(data_path, "train"), train_tf)
    ts_ds = WeatherDS(os.path.join(data_path, "test"), val_tf)
    # 从训练集切分验证集
    vs_ = max(1, int(0.15 * len(tr_ds)))
    ts_ = len(tr_ds) - vs_
    tr_ds, vl_ds = torch.utils.data.random_split(
        tr_ds, [ts_, vs_], generator=torch.Generator().manual_seed(42))
    vl_ds.dataset.transform = val_tf  # type: ignore
    print(f"\nTrain={len(tr_ds)} Val={len(vl_ds)} Test={len(ts_ds)}")

    tl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vl = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Step 3: 构建模型(加载预训练权重)
    model = build_model(pretrained_path=PRETRAINED_MODEL)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Step 4: 训练
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    for p in model.parameters(): p.requires_grad = True
    hp = [p for n,p in model.named_parameters() if "classifier" in n]
    bp = [p for n,p in model.named_parameters() if "classifier" not in n]
    opt = optim.AdamW([{"params":hp,"lr":LR},{"params":bp,"lr":FINE_LR}], weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_f1, best_ep, es_cnt = 0.0, 0, 0
    t0 = time.time()
    for ep in range(NUM_EPOCHS):
        model.train(); tl_ = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            ma = torch.rand(1).item() < MIXUP_P
            if ma:
                if torch.rand(1).item() < 0.6: x, ya, yb, lam = mixup(x, y, MIXUP_A)
                else: x, ya, yb, lam = cutmix(x, y, CUTMIX_A)
            opt.zero_grad(set_to_none=True)
            if scaler:
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

        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(DEVICE)); vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.cpu().numpy())
        acc = accuracy_score(vt, vp); f1 = f1_score(vt, vp, average="macro")
        print(f"FT E{ep+1:3d}/{NUM_EPOCHS} | loss={tl_/len(tl):.4f} | acc={acc:.4f} | f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_ep, es_cnt = f1, ep, 0
            torch.save({"epoch":ep,"model":model.state_dict(),"f1":f1}, OUTPUT_MODEL)
            print(f"  >>> Best (F1={best_f1:.4f})")
        else:
            es_cnt += 1
            if es_cnt >= 10:
                print(f"EarlyStop @ E{ep+1}, Best F1={best_f1:.4f} @ E{best_ep+1}")
                break

    print(f"\nDone: {time.time()-t0:.0f}s | Best Val F1={best_f1:.4f} @ E{best_ep+1}")

    # Step 5: TTA测试
    print(f"\n=== TTA Test ===")
    ckpt = torch.load(OUTPUT_MODEL, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    test_files = []
    for cls in TRAIN_CLASSES:
        d = os.path.join(data_path, "test", cls)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                    test_files.append((os.path.join(d,f), TRAIN_CLASSES.index(cls)))

    preds, tgts = [], []
    for p, l in test_files:
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(0,0,0))
        prob = tta_predict(model, img)
        preds.append(prob.argmax().item()); tgts.append(l)

    acc = accuracy_score(tgts, preds); f1m = f1_score(tgts, preds, average="macro")
    per = f1_score(tgts, preds, average=None); cm = confusion_matrix(tgts, preds)
    print(f"TTA: Acc={acc:.4f} F1={f1m:.4f}")
    for i, cls in enumerate(TRAIN_CLASSES):
        print(f"  {cls}: F1={per[i]:.4f}")
    print(f"  c→s={cm[0][2]} s→c={cm[2][0]}")

    # 清理临时数据
    shutil.rmtree(data_path)
    print(f"\nModel saved: {OUTPUT_MODEL}")
