"""
智海Mo平台部署 - Handler函数
应用名称: 天气图片分类 (Weather Classification)
功能: 识别图片中的天气类型 (晴天/雨天/阴天/雪天)

部署时，Mo平台会自动识别 handle 函数的输入输出参数生成 app_spec.yml
"""

import os
import time
import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# ======== 全局配置 ========
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["sunny", "rainy", "cloudy", "snowy"]
DEVICE = torch.device("cpu")  # 部署环境可能无GPU，用CPU保证兼容

# 图片预处理
_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 全局模型实例 (延迟加载)
_model = None


def build_model():
    """构建模型（与训练时保持一致）"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def get_model(model_path="best_model.pth"):
    """单例模式加载模型（避免每次推理都重新加载）"""
    global _model
    if _model is None:
        _model = build_model()
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            _model.load_state_dict(state_dict, strict=False)
        _model.eval()
    return _model


def preprocess_image(image_input):
    """
    预处理输入图片（支持多种输入格式）

    Args:
        image_input: 可以是:
            - 文件路径 (str)
            - Base64编码的图片字符串 (str)
            - 二进制数据 (bytes)
            - PIL Image对象
    Returns:
        torch.Tensor: (1, 3, H, W)
    """
    if isinstance(image_input, str):
        # 尝试作为文件路径
        if os.path.exists(image_input):
            image = Image.open(image_input).convert("RGB")
        else:
            # 尝试作为Base64解码
            try:
                image_bytes = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                raise ValueError(
                    f"无法解析image_input: 不是有效的文件路径或Base64字符串"
                )
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"不支持的图片输入格式: {type(image_input)}")

    tensor = _transform(image).unsqueeze(0)
    return tensor, image


def handle(image_input, model_path="best_model.pth"):
    """
    天气分类应用的Handle函数 (Mo平台部署入口)

    --- 输入参数 ---
    image_input: str - 待分类的图片，支持以下格式:
                     (1) 图片文件路径
                     (2) Base64编码的图片字符串
                     (3) 二进制图片数据
    model_path: str  - 模型权重文件路径 (默认: best_model.pth)

    --- 输出 ---
    dict: {
        "prediction": str,          - 预测的天气类别
        "prediction_cn": str,       - 中文天气类别
        "confidence": float,        - 置信度 (0~1)
        "probabilities": dict,      - 各类别概率
        "all_predictions": list,    - 各类别预测详情
        "inference_time_ms": float, - 推理耗时(毫秒)
        "image_size": str,          - 输入图片尺寸
    }
    """
    start_time = time.time()

    # 1. 加载模型
    model = get_model(model_path)

    # 2. 预处理
    tensor, pil_image = preprocess_image(image_input)

    # 3. 推理
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    inference_time = (time.time() - start_time) * 1000

    # 4. 构建结果
    pred_idx = int(np.argmax(probs))
    cn_map = {
        "sunny": "晴天",
        "rainy": "雨天",
        "cloudy": "阴天",
        "snowy": "雪天",
    }

    all_predictions = []
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        all_predictions.append({
            "class_id": i,
            "class_name": name,
            "class_name_cn": cn_map.get(name, name),
            "probability": round(float(prob), 4),
            "is_prediction": i == pred_idx,
        })

    # 按概率降序排列
    all_predictions.sort(key=lambda x: x["probability"], reverse=True)

    result = {
        "prediction": CLASS_NAMES[pred_idx],
        "prediction_cn": cn_map.get(CLASS_NAMES[pred_idx], CLASS_NAMES[pred_idx]),
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(CLASS_NAMES, probs)
        },
        "all_predictions": all_predictions,
        "inference_time_ms": round(inference_time, 2),
        "image_size": f"{pil_image.size[0]}x{pil_image.size[1]}",
    }

    return result


# ======== 本地测试 ========
if __name__ == "__main__":
    print("天气分类应用 Handler 已就绪")
    print(f"支持的四类天气: {CLASS_NAMES}")
    print(f"模型路径: best_model.pth")

    # 如果有测试图片
    test_img = "test_sample.jpg"
    if os.path.exists(test_img):
        result = handle(test_img)
        print(f"\n测试结果:")
        print(f"  预测: {result['prediction']} ({result['prediction_cn']})")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"  推理时间: {result['inference_time_ms']}ms")
        print(f"  各类概率:")
        for p in result["all_predictions"]:
            marker = " <--" if p["is_prediction"] else ""
            print(f"    {p['class_name_cn']}: {p['probability']:.2%}{marker}")
    else:
        print(f"提示: 将测试图片放到当前目录作为 {test_img} 可进行测试")
