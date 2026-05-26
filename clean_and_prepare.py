"""
数据清洗 + 重采样脚本
从源数据集 69f46e75dbb43ba9e05483c1 取四类天气数据:
  cloudy -> cloudy (阴天)
  rainy  -> rain   (雨天)
  sunny  -> sunny  (晴天)
  snow   -> snow   (下雪)

清洗策略:
  1. 损坏文件 (无法打开)
  2. 过小图片 (< 100px)
  3. 完全重复 (MD5)
  4. 过度模糊 (Laplacian variance)
  5. 近纯色图片 (RGB std 极低)
  6. 模型置信度检测 (低置信度/预测与标签不符)

输出:
  data/weather_clean/train/  800/类
  data/weather_clean/test/   200/类
  output/removed_images/     被剔除的图片(按原因+类别)
"""

import os
import sys
import hashlib
import shutil
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict
# tqdm not required

# ============ 配置 ============
SRC_BASE = "c:/Users/19709/Desktop/睿抗/69f46e75dbb43ba9e05483c1/weather_classification"
DST_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather_clean")
REMOVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "removed_images")

# 四类映射: 源目录名 -> 目标类名
CLASS_MAP = {
    "cloudy": "cloudy",
    "rainy":  "rain",
    "sunny":  "sunny",
    "snow":   "snow",
}

TRAIN_PER_CLASS = 800
TEST_PER_CLASS = 200
IMAGE_SIZE = 224

# 清洗阈值
MIN_RESOLUTION = 100       # 最小边长
BLUR_THRESHOLD = 100.0     # Laplacian方差阈值 (低于此值=模糊)
LOW_STD_THRESHOLD = 15.0   # RGB标准差阈值 (低于此值=近纯色)

# 随机种子
SEED = 42
np.random.seed(SEED)

os.makedirs(REMOVED_DIR, exist_ok=True)

total_removed = defaultdict(list)  # reason -> [(src_path, class, info)]


def compute_md5(filepath):
    """计算文件MD5"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_laplacian_variance(img_array):
    """计算Laplacian方差 (模糊度指标)"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_rgb_std(img_array):
    """计算RGB各通道平均标准差 (色彩多样性指标)"""
    return np.mean([np.std(img_array[:, :, c]) for c in range(3)])


def is_valid_image(filepath):
    """检查图片是否可正常打开"""
    try:
        img = Image.open(filepath)
        img.verify()
        # verify后需要重新打开才能读数据
        img = Image.open(filepath)
        img.load()
        return True, img.size
    except Exception:
        return False, None


def save_removed(src_path, target_cls, reason, info=""):
    """将剔除图片复制到review目录"""
    dst_dir = os.path.join(REMOVED_DIR, reason, target_cls)
    os.makedirs(dst_dir, exist_ok=True)
    fname = os.path.basename(src_path)
    # 加info防重名
    if info:
        name, ext = os.path.splitext(fname)
        fname = f"{name}_{info}{ext}"
    dst_path = os.path.join(dst_dir, fname)
    try:
        shutil.copy2(src_path, dst_path)
    except Exception:
        pass
    total_removed[reason].append((src_path, target_cls, info))


def clean_and_prepare():
    print("=" * 60)
    print("数据清洗与重采样")
    print(f"源数据集: {SRC_BASE}")
    print(f"目标目录: {DST_BASE}")
    print(f"剔除目录: {REMOVED_DIR}")
    print(f"类别映射: {CLASS_MAP}")
    print(f"每类: train={TRAIN_PER_CLASS}, test={TEST_PER_CLASS}")
    print("=" * 60)

    # ====== 阶段1: 扫描所有图片 ======
    print("\n[阶段1] 扫描源数据集...")
    all_images = {cls: [] for cls in CLASS_MAP.values()}

    for src_cls, dst_cls in CLASS_MAP.items():
        src_dir = os.path.join(SRC_BASE, src_cls)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: {src_dir} 不存在!")
            continue

        files = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        for f in files:
            all_images[dst_cls].append(os.path.join(src_dir, f))

        print(f"  {src_cls} -> {dst_cls}: {len(files)} 张")

    # ====== 阶段2: 损坏文件检测 ======
    print("\n[阶段2] 损坏文件检测...")
    corrupt_count = 0
    for cls, paths in all_images.items():
        valid_paths = []
        for p in paths:
            ok, size = is_valid_image(p)
            if not ok:
                save_removed(p, cls, "corrupt")
                corrupt_count += 1
            elif size[0] < MIN_RESOLUTION or size[1] < MIN_RESOLUTION:
                save_removed(p, cls, "too_small", f"{size[0]}x{size[1]}")
                corrupt_count += 1
            else:
                valid_paths.append(p)
        all_images[cls] = valid_paths
    print(f"  剔除 {corrupt_count} 张")

    # ====== 阶段3: 重复文件检测 ======
    print("\n[阶段3] 重复文件检测 (MD5)...")
    md5_map = {}  # md5 -> (first_path, class)
    dup_count = 0
    for cls, paths in all_images.items():
        unique_paths = []
        for p in paths:
            md5 = compute_md5(p)
            if md5 in md5_map:
                save_removed(p, cls, "duplicate", f"dup_of_{os.path.basename(md5_map[md5][0])}")
                dup_count += 1
            else:
                md5_map[md5] = (p, cls)
                unique_paths.append(p)
        all_images[cls] = unique_paths
    print(f"  剔除 {dup_count} 张")

    # ====== 阶段4: 模糊/低质量检测 ======
    print("\n[阶段4] 模糊与低质量检测...")
    blur_count = 0
    lowstd_count = 0

    for cls, paths in all_images.items():
        clean_paths = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img_array = np.array(img)

                # 模糊检测
                lap_var = compute_laplacian_variance(img_array)
                if lap_var < BLUR_THRESHOLD:
                    save_removed(p, cls, "blurry", f"lapvar={lap_var:.1f}")
                    blur_count += 1
                    continue

                # 近纯色检测
                rgb_std = compute_rgb_std(img_array)
                if rgb_std < LOW_STD_THRESHOLD:
                    save_removed(p, cls, "low_contrast", f"std={rgb_std:.1f}")
                    lowstd_count += 1
                    continue

                clean_paths.append(p)
            except Exception:
                save_removed(p, cls, "processing_error")
                continue

        all_images[cls] = clean_paths
    print(f"  模糊剔除: {blur_count} 张")
    print(f"  低对比度剔除: {lowstd_count} 张")

    # ====== 阶段5: 模型置信度检测 (跳过——旧模型类名不同) ======
    print("\n[阶段5] 模型置信度检测 (跳过——旧模型用 sunrise，新数据用 snow，类名不兼容)")

    # ====== 阶段6: 汇总 & 采样 ======
    print("\n[阶段6] 汇总与采样...")
    print(f"\n{'类别':<10} {'剩余':<8} {'需Train':<10} {'需Test':<10}")
    print("-" * 40)

    # 清理目标目录
    if os.path.exists(DST_BASE):
        shutil.rmtree(DST_BASE)

    for cls in CLASS_MAP.values():
        available = len(all_images[cls])
        needed = TRAIN_PER_CLASS + TEST_PER_CLASS
        if available < needed:
            print(f"  {cls:<10} {available:<8} (不足! 缺 {needed - available} 张)")
        else:
            print(f"  {cls:<10} {available:<8} {TRAIN_PER_CLASS:<10} {TEST_PER_CLASS:<10}")

        # 随机选择
        selected = np.random.choice(all_images[cls], size=min(needed, available), replace=False)

        train_imgs = selected[:TRAIN_PER_CLASS]
        test_imgs = selected[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]

        for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(DST_BASE, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

    # ====== 阶段7: 剔除汇总报告 ======
    print("\n" + "=" * 60)
    print("剔除汇总报告")
    print("=" * 60)

    total_removed_count = 0
    for reason, items in sorted(total_removed.items()):
        reason_dir = os.path.join(REMOVED_DIR, reason)
        count = len(items)
        total_removed_count += count
        # 按类别统计
        cls_counts = defaultdict(int)
        for _, cls, _ in items:
            cls_counts[cls] += 1
        cls_str = " ".join(f"{c}={n}" for c, n in sorted(cls_counts.items()))
        print(f"  [{reason}] {count} 张  ({cls_str})")
        print(f"         保存位置: {reason_dir}")

    print(f"\n总计剔除: {total_removed_count} 张")
    print(f"保留用于训练: {TRAIN_PER_CLASS * 4} 张")
    print(f"保留用于测试: {TEST_PER_CLASS * 4} 张")

    # 保存剔除清单
    manifest_path = os.path.join(REMOVED_DIR, "removed_manifest.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("被剔除图片清单\n")
        f.write("=" * 60 + "\n\n")
        for reason, items in sorted(total_removed.items()):
            f.write(f"\n## {reason} ({len(items)} 张)\n")
            for src_path, cls, info in items:
                f.write(f"  [{cls}] {os.path.basename(src_path)}")
                if info:
                    f.write(f"  -- {info}")
                f.write("\n")

    print(f"\n剔除清单: {manifest_path}")
    print("完成!")


def run_model_check(all_images):
    """使用已有最佳模型检测可能错标的图片"""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "output", "best_model_w800.pth")
    if not os.path.exists(model_path):
        print("  跳过 (无已有模型)")
        return 0

    try:
        import torch
        from torchvision import transforms, models

        device = torch.device("cpu")
        # 加载模型
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 4),
        )
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)

        val_transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        # 旧类别名映射: ["cloudy", "rain", "shine", "sunrise"] -> 新
        old_names = ["cloudy", "rain", "shine", "sunrise"]
        name_to_idx = {n: i for i, n in enumerate(old_names)}

        # 源类名 -> 旧模型类名
        # cloudy->cloudy, rainy->rain, sunny->shine, snow->sunrise
        cls_to_old = {
            "cloudy": "cloudy",
            "rain": "rain",
            "sunny": "shine",
            "snow": "sunrise",
        }

        removed_count = 0
        threshold = 0.5  # 置信度阈值

        for cls, paths in all_images.items():
            old_cls = cls_to_old.get(cls)
            if old_cls not in name_to_idx:
                continue
            expected_idx = name_to_idx[old_cls]
            clean_paths = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = val_transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                        conf = conf.item()
                        pred = pred.item()
                    # 低置信度 或 预测与标签不符
                    if conf < threshold:
                        save_removed(p, cls, "low_confidence",
                                     f"conf={conf:.3f}_pred={old_names[pred]}")
                        removed_count += 1
                    elif pred != expected_idx:
                        save_removed(p, cls, "mislabeled_by_model",
                                     f"pred={old_names[pred]}_conf={conf:.3f}")
                        removed_count += 1
                    else:
                        clean_paths.append(p)
                except Exception:
                    clean_paths.append(p)
            all_images[cls] = clean_paths

        return removed_count

    except Exception as e:
        print(f"  模型检测出错: {e}")
        return 0


if __name__ == "__main__":
    clean_and_prepare()
