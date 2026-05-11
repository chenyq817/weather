"""
推理与提交: 支持TTA、TorchScript编译、批量推理
"""

import os
import time
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from config import (
    MODEL_NAME, NUM_CLASSES, CLASS_NAMES, CKPT_DIR, DEVICE,
    TEST_DIR, IMAGE_SIZE, TTA_TIMES,
    USE_TORCHSCRIPT, INFERENCE_BATCH_SIZE, OUTPUT_DIR,
)
from model import get_model
from dataset import get_val_transforms, get_tta_transforms, create_inference_loader


def load_trained_model(ckpt_path=None, torchscript=True):
    """加载训练好的模型"""
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

    model = get_model(MODEL_NAME, pretrained=False)
    model = model.to(DEVICE)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"  Best epoch metrics: {checkpoint.get('metrics', 'N/A')}")
    else:
        print(f"Warning: checkpoint not found at {ckpt_path}, using random weights")

    model.eval()

    # TorchScript 编译加速推理
    if torchscript and USE_TORCHSCRIPT:
        try:
            example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            model = torch.jit.trace(model, example_input)
            print("  TorchScript compiled successfully")
        except Exception as e:
            print(f"  TorchScript compilation skipped: {e}")

    return model


@torch.no_grad()
def predict_single_image(model, image_path, transform):
    """单张图片推理"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    output = model(tensor)
    probs = F.softmax(output, dim=1)
    return probs.cpu().numpy()[0]


@torch.no_grad()
def predict_batch(model, dataloader):
    """批量推理"""
    all_probs = []
    all_preds = []

    for images, _ in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
    )


@torch.no_grad()
def predict_with_tta(model, image_path, tta_transforms):
    """测试时增强(TTA)推理: 多个变换的结果取平均"""
    image = Image.open(image_path).convert("RGB")
    all_probs = []

    for transform in tta_transforms:
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        all_probs.append(probs.cpu().numpy())

    avg_probs = np.mean(all_probs, axis=0)[0]
    return avg_probs


@torch.no_grad()
def predict_directory(model, image_dir, use_tta=True):
    """对整个目录进行推理并生成提交文件"""
    start_time = time.time()

    # 获取所有图片路径
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ])

    if not image_files:
        # 尝试递归查找
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in image_exts:
                    image_files.append(os.path.relpath(
                        os.path.join(root, f), image_dir
                    ))
        image_files = sorted(image_files)

    print(f"Found {len(image_files)} images in {image_dir}")

    results = []
    transform = get_val_transforms()
    tta_list = get_tta_transforms() if use_tta and TTA_TIMES > 0 else None
    num_tta = len(tta_list) if tta_list else 1

    for idx, fname in enumerate(image_files):
        fpath = os.path.join(image_dir, fname)

        if tta_list:
            probs = predict_with_tta(model, fpath, tta_list)
        else:
            probs = predict_single_image(model, fpath, transform)

        pred_class = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_class]
        confidence = float(probs[pred_class])

        results.append({
            "filename": fname,
            "prediction": pred_label,
            "class_id": pred_class,
            "confidence": confidence,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(image_files)} images...")

    elapsed = time.time() - start_time
    avg_time = elapsed / len(image_files) * 1000 if image_files else 0
    print(f"\nInference complete!")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Images: {len(image_files)}")
    print(f"  Avg per image: {avg_time:.2f}ms")
    print(f"  TTA: {'enabled' if tta_list else 'disabled'} ({num_tta} views)")

    return results


def save_submission(results, output_path=None):
    """保存提交文件 (CSV格式)"""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "submission.csv")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for r in results:
            writer.writerow([r["filename"], r["prediction"]])

    print(f"Submission saved to: {output_path}")
    return output_path


def print_class_distribution(results):
    """打印预测类别分布"""
    from collections import Counter
    counts = Counter(r["prediction"] for r in results)
    total = len(results)
    print("\nPrediction distribution:")
    for cls_name in CLASS_NAMES:
        c = counts.get(cls_name, 0)
        print(f"  {cls_name}: {c} ({c/total*100:.1f}%)")


def inference_and_submit(ckpt_path=None, test_dir=None, use_tta=True):
    """一键推理 + 生成提交文件"""
    if test_dir is None:
        test_dir = TEST_DIR

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model = load_trained_model(ckpt_path)
    results = predict_directory(model, test_dir, use_tta=use_tta)
    print_class_distribution(results)

    output_path = save_submission(results)
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="天气分类推理")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="模型权重路径")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="测试集目录")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="启用测试时增强")
    parser.add_argument("--no-tta", action="store_false", dest="tta",
                        help="禁用测试时增强")
    args = parser.parse_args()

    inference_and_submit(
        ckpt_path=args.ckpt,
        test_dir=args.test_dir,
        use_tta=args.tta,
    )
