"""
天气分类部署 Handler - ConvNeXt-Tiny 双模型 TTA 集成
类别: cloudy(阴天) / rain(雨天) / sunny(晴天) / snow(雪天)
"""
import os, time, base64, io
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models

IMAGE_SIZE = 260; NUM_CLASSES = 4
CLASS_NAMES = ["cloudy", "rain", "sunny", "snow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TTA_SIZE = int(IMAGE_SIZE * 1.14)
to_tensor = transforms.ToTensor()
norm_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Cache
_models = None

def build_model():
    model = models.convnext_tiny(weights=None)
    inf = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.LayerNorm(inf, eps=1e-6),
        nn.Dropout(0.4), nn.Linear(inf, 256),
        nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES),
    )
    return model

def load_models(model_paths=None):
    """加载双模型用于集成推理"""
    global _models
    if _models is not None:
        return _models

    if model_paths is None:
        model_paths = ["best_model_v10_convnext.pth", "best_model_v10_convnext_s456.pth"]

    _models = []
    for mp in model_paths:
        m = build_model()
        if os.path.exists(mp):
            ckpt = torch.load(mp, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model", ckpt))
        m.eval().to(DEVICE)
        _models.append(m)
    return _models

def tta_predict(model, img):
    img_r = transforms.Resize(TTA_SIZE)(img)
    crops = transforms.FiveCrop(IMAGE_SIZE)(img_r)
    probs = []
    for crop in crops:
        for flip in [False, True]:
            c = transforms.RandomHorizontalFlip(p=1.0)(crop) if flip else crop
            t = norm_t(to_tensor(c)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs.append(torch.softmax(model(t), 1))
    return torch.cat(probs).mean(0, keepdim=True)

def preprocess(image_input):
    if isinstance(image_input, str):
        if os.path.exists(image_input):
            img = Image.open(image_input).convert("RGB")
        else:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")
            except:
                raise ValueError("无法解析image_input")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"不支持的类型: {type(image_input)}")
    return img

def handle(image_input, model_path="best_model_v10_convnext.pth"):
    """
    天气分类 Handle 函数

    输入:
        image_input: 图片路径 / Base64 / bytes / PIL Image
        model_path:  模型路径(单模型) 或 模型路径列表(双模型集成)
    输出:
        dict: prediction, confidence, probabilities, inference_time_ms
    """
    t0 = time.time()

    # 预处理
    img = preprocess(image_input)

    # 加载模型 - 支持单模型或双模型
    if isinstance(model_path, list) and len(model_path) == 2:
        models_list = load_models(model_path)
        # 双模型集成
        p1 = tta_predict(models_list[0], img)
        p2 = tta_predict(models_list[1], img)
        probs = ((p1 + p2) / 2).cpu().numpy()[0]
    else:
        if isinstance(model_path, list):
            model_path = model_path[0]
        models_list = load_models([model_path])
        probs = tta_predict(models_list[0], img).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    elapsed = (time.time() - t0) * 1000

    cn_map = {"cloudy": "阴天", "rain": "雨天", "sunny": "晴天", "snow": "雪天"}

    all_preds = []
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        all_preds.append({
            "class_id": i, "class_name": name,
            "class_name_cn": cn_map.get(name, name),
            "probability": round(float(p), 4),
            "is_prediction": i == pred_idx,
        })
    all_preds.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "prediction_cn": cn_map.get(CLASS_NAMES[pred_idx], CLASS_NAMES[pred_idx]),
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        "all_predictions": all_preds,
        "inference_time_ms": round(elapsed, 2),
        "image_size": f"{img.size[0]}x{img.size[1]}",
    }


if __name__ == "__main__":
    print(f"天气分类 Handler (ConvNeXt-Tiny + TTA) 已就绪")
    print(f"Device: {DEVICE} | Classes: {CLASS_NAMES}")
    print(f"双模型集成: best_model_v10_convnext.pth + best_model_v10_convnext_s456.pth")

    test_img = "test_sample.jpg"
    if os.path.exists(test_img):
        result = handle(test_img)
        print(f"\n预测: {result['prediction_cn']} (置信度: {result['confidence']:.2%})")
        for p in result["all_predictions"]:
            marker = " <--" if p["is_prediction"] else ""
            print(f"  {p['class_name_cn']}: {p['probability']:.2%}{marker}")
        print(f"推理耗时: {result['inference_time_ms']}ms")
    else:
        print(f"提示: 放一张测试图作为 {test_img} 可测试")
