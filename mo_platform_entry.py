"""
智海Mo平台适配入口
根据平台提供的API接口编写

常见平台接口模式:
  1. 平台提供 train_dataset, val_dataset, test_dataset 对象
  2. 平台提供 DataLoader 或直接调用 train() / predict() 函数
  3. 平台通过命令行参数传入数据路径
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score

# ============================================================
# 智海Mo平台标准接口
# 请根据实际比赛平台提供的模板调整
# ============================================================

# ---- 配置 ----
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
NUM_CLASSES = 4
CLASS_NAMES = ["cloudy", "rain", "shine", "sunrise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 数据增强 ----
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PlatformDataset(Dataset):
    """适配智海Mo平台的数据集包装器"""

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array or list of PIL Images
            labels: numpy array or list of int
            transform: torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # 如果是 numpy array, 转为 PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img

        if self.transform:
            img = self.transform(img)

        return img, label


def build_model():
    """构建 EfficientNet-B0 模型"""
    import torchvision.models as models
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ============================================================
# 平台标准训练函数 (可能被平台自动调用)
# ============================================================

def train(train_dataset, val_dataset=None):
    """
    智海Mo平台训练接口

    Args:
        train_dataset: 平台提供的训练数据集
        val_dataset: 平台提供的验证数据集 (可能为None)
    Returns:
        model: 训练好的模型
    """
    # 包装数据
    if hasattr(train_dataset, '__getitem__'):
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True,
        )
    else:
        raise ValueError("train_dataset 格式不支持")

    val_loader = None
    if val_dataset is not None and hasattr(val_dataset, '__getitem__'):
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True,
        )

    # 构建模型
    model = build_model()

    # 损失函数 (Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()

        # ---- Validate ----
        if val_loader:
            model.eval()
            val_preds, val_targets = [], []

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(targets.numpy())

            val_f1 = f1_score(val_targets, val_preds, average="macro")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

    return model


# ============================================================
# 平台标准预测函数 (可能被平台自动调用)
# ============================================================

def predict(test_dataset, model_path="best_model.pth"):
    """
    智海Mo平台推理接口

    Args:
        test_dataset: 平台提供的测试数据集
        model_path: 模型权重路径
    Returns:
        predictions: list of class indices
    """
    # 加载模型
    model = build_model()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()

    # 包装数据
    if hasattr(test_dataset, '__getitem__'):
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False,
            num_workers=2, pin_memory=True,
        )
    else:
        raise ValueError("test_dataset 格式不支持")

    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())

    return all_preds


# ============================================================
# 本地测试
# ============================================================

if __name__ == "__main__":
    print(f"智海Mo平台适配脚本已就绪")
    print(f"Device: {DEVICE}")
    print(f"类别: {CLASS_NAMES}")

    # 测试模型构建
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params/1e6:.2f}M")

    # 测试推理速度
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        # 计时
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

    print(f"推理速度: {elapsed/100*1000:.2f}ms/image (avg over 100 runs)")
