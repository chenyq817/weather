"""
数据集加载与增强管线
支持: 多视角天气图片分类, 四类天气(晴天/雨天/阴天/雪天)
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import autoaugment, functional as TF

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE,
    NUM_WORKERS, AUGMENTATION,
)


def get_train_transforms():
    """训练集数据增强管线"""
    aug = AUGMENTATION

    # 基础策略列表
    transform_list = [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=aug["random_crop_scale"]),
        transforms.RandomHorizontalFlip(p=aug["random_h_flip"]),
        transforms.RandomVerticalFlip(p=aug["random_v_flip"]),
        transforms.RandomRotation(aug["random_rotation"]),
        transforms.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
            hue=aug["color_jitter"]["hue"],
        ),
    ]

    # 自动增强策略 (AutoAugment / RandAugment / TrivialAugmentWide)
    if aug.get("auto_augment"):
        transform_list.append(autoaugment.TrivialAugmentWide())

    # 转Tensor + 标准化
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    # 随机擦除
    if aug.get("random_erasing_p", 0) > 0:
        transform_list.append(
            transforms.RandomErasing(p=aug["random_erasing_p"], mode='pixel')
        )

    return transforms.Compose(transform_list)


def get_val_transforms():
    """验证集/测试集管线"""
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms():
    """测试时增强(TTA) - 多个crop+翻转组合"""
    tta_list = []

    # 中心裁剪 + 水平翻转
    base = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    tta_list.append(base)
    tta_list.append(transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ]))

    # 四个角 + 中心 crops
    corner_crops = [
        transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.FiveCrop(IMAGE_SIZE),
            transforms.Lambda(lambda crops: crops[i]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        for i in range(5)
    ]
    tta_list.extend(corner_crops)

    return tta_list


class WeatherDataset(Dataset):
    """通用天气数据集类
    支持两种数据组织方式:
    1. 按类别文件夹: data/train/sunny/*.jpg, data/train/rainy/*.jpg ...
    2. 单一文件夹+标签文件: data/train/*.jpg + train_labels.csv
    """

    def __init__(self, root_dir, transform=None, label_file=None):
        """
        Args:
            root_dir: 数据根目录
            transform: torchvision transforms
            label_file: 标签CSV文件路径 (可选, 若为None则按文件夹结构读取)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        if label_file and os.path.exists(label_file):
            # 方式2: 从CSV读取标签
            self._load_from_csv(label_file)
        else:
            # 方式1: 从文件夹结构读取
            self._load_from_folders()

    def _load_from_folders(self):
        """按ImageFolder方式加载"""
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name]
                    ))

    def _load_from_csv(self, label_file):
        """从CSV标签文件加载 (格式: filename,label)"""
        import csv
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    fname, label = row[0].strip(), row[1].strip()
                    fpath = os.path.join(self.root_dir, fname)
                    if os.path.exists(fpath):
                        # 标签可以是字符串或数字
                        if label.isdigit():
                            label_idx = int(label)
                        else:
                            label_idx = self.class_to_idx.get(label.lower(), -1)
                        if label_idx >= 0:
                            self.samples.append((fpath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 处理图片读取异常
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError):
            # 损坏图片返回黑色图
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def mixup_data(x, y, alpha=0.2):
    """MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * y[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.2):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    # 生成随机裁剪区域
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bw = int(W * np.sqrt(1 - lam))
    bh = int(H * np.sqrt(1 - lam))
    x0 = max(cx - bw // 2, 0)
    y0 = max(cy - bh // 2, 0)
    x1 = min(cx + bw // 2, W)
    y1 = min(cy + bh // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y0:y1, x0:x1] = x[index, :, y0:y1, x0:x1]

    # 实际混合比例
    lam = 1 - ((x1 - x0) * (y1 - y0) / (H * W))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_dataloaders():
    """创建训练/验证/测试的DataLoader"""
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # 训练集
    if os.path.exists(TRAIN_DIR):
        train_dataset = WeatherDataset(TRAIN_DIR, transform=get_train_transforms())

    # 验证集
    if os.path.exists(VAL_DIR):
        val_dataset = WeatherDataset(VAL_DIR, transform=get_val_transforms())

    # 如果无独立验证集，从训练集切分
    if val_dataset is None and train_dataset is not None:
        from torch.utils.data import random_split
        val_size = max(1, int(0.15 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        # val用测试变换
        val_dataset.dataset.transform = get_val_transforms()

    # 测试集
    if os.path.exists(TEST_DIR):
        test_dataset = WeatherDataset(TEST_DIR, transform=get_val_transforms())

    dataloaders = {}

    if train_dataset:
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,  # mixup/cutmix需要对齐batch
        )

    if val_dataset:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    if test_dataset:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    return dataloaders


def create_inference_loader(image_dir, batch_size=128):
    """推理专用DataLoader"""
    dataset = WeatherDataset(image_dir, transform=get_val_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
