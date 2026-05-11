"""
从大数据集中抽取子集用于本地训练和测试
选取4类: sunny, rainy, cloudy, snow
每类: train=400, val=100, test=100 (共2400张)
"""

import os
import shutil
import random

# 路径配置
SRC_BASE = "c:/Users/19709/Desktop/睿抗/69f46e75dbb43ba9e05483c1/weather_classification"
DST_BASE = "c:/Users/19709/Desktop/睿抗/weather_classification/data"

# 选取的4个类别
CLASSES = ["cloudy", "rain", "shine", "sunrise"]

# 每个类别抽样数量
TRAIN_PER_CLASS = 800
VAL_PER_CLASS = 100
TEST_PER_CLASS = 100

# 随机种子保证可复现
random.seed(42)

for cls in CLASSES:
    src_dir = os.path.join(SRC_BASE, cls)
    if not os.path.isdir(src_dir):
        print(f"Warning: {src_dir} not found, skipping {cls}")
        continue

    # 获取所有图片
    all_images = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

    # 随机打乱
    random.shuffle(all_images)

    # 按比例分配
    total_needed = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
    selected = all_images[:total_needed]

    train_imgs = selected[:TRAIN_PER_CLASS]
    val_imgs = selected[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
    test_imgs = selected[TRAIN_PER_CLASS + VAL_PER_CLASS:]

    # 创建目标目录
    for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        dst_dir = os.path.join(DST_BASE, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(src_dir, img)
            dst = os.path.join(dst_dir, img)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copying {img}: {e}")

        count = len(os.listdir(dst_dir))
        print(f"  {split}/{cls}: {count} images")

    print(f"[{cls}] Total: {total_needed} images selected from {len(all_images)}")

# 打印汇总
print("\n" + "=" * 50)
print("Dataset Summary")
print("=" * 50)
for split in ["train", "val", "test"]:
    total = 0
    for cls in CLASSES:
        d = os.path.join(DST_BASE, split, cls)
        if os.path.isdir(d):
            total += len(os.listdir(d))
    print(f"{split}: {total} images")

total_all = 0
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        d = os.path.join(DST_BASE, split, cls)
        if os.path.isdir(d):
            total_all += len(os.listdir(d))
print(f"Total: {total_all} images")
