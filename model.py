"""
模型定义: 多种轻量级分类模型, 支持迁移学习
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, PRETRAINED, DROPOUT_RATE


def get_model(model_name="efficientnet_b0", pretrained=True):
    """获取指定模型

    可选模型:
        efficientnet_b0/b1/b2  - EfficientNet系列 (推荐)
        mobilenet_v3_small/large - MobileNetV3系列 (最快)
        resnet34/50              - ResNet系列
        densenet121              - DenseNet系列 (最准但慢)
        convnext_tiny            - ConvNeXt系列
    """
    builder = {
        "efficientnet_b0": _build_efficientnet_b0,
        "efficientnet_b1": _build_efficientnet_b1,
        "efficientnet_b2": _build_efficientnet_b2,
        "mobilenet_v3_small": _build_mobilenet_v3_small,
        "mobilenet_v3_large": _build_mobilenet_v3_large,
        "resnet34": _build_resnet34,
        "resnet50": _build_resnet50,
        "densenet121": _build_densenet121,
        "convnext_tiny": _build_convnext_tiny,
    }

    if model_name not in builder:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(builder.keys())}")

    return builder[model_name](pretrained)


class WeatherModel(nn.Module):
    """天气分类模型基类"""

    def __init__(self, backbone, classifier, feature_dim=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


def _build_efficientnet_b0(pretrained=True):
    """EfficientNet-B0: 5.3M参数, 97.4%测试准确率
    推荐首选: 最佳精度/速度平衡
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_efficientnet_b1(pretrained=True):
    """EfficientNet-B1: 7.8M参数, 精度稍高于B0"""
    if pretrained:
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b1(weights=None)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_efficientnet_b2(pretrained=True):
    """EfficientNet-B2: 9.1M参数"""
    if pretrained:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b2(weights=None)

    in_features = model.classifier[1].in_features  # 1408
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_mobilenet_v3_large(pretrained=True):
    """MobileNetV3-Large: 5.4M参数, 推理最快
    推荐: 对推理速度要求极高时使用
    """
    if pretrained:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_large(weights=None)

    in_features = model.classifier[3].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def _build_mobilenet_v3_small(pretrained=True):
    """MobileNetV3-Small: 2.5M参数, 超轻量"""
    if pretrained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def _build_resnet34(pretrained=True):
    """ResNet34: 21M参数, 经典基线"""
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_resnet50(pretrained=True):
    """ResNet50: 25M参数"""
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_densenet121(pretrained=True):
    """DenseNet121: 8M参数, 98.4%最佳准确率(文献)
    精度最高, 但推理速度比EfficientNet慢约30%
    """
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet121(weights=None)

    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _build_convnext_tiny(pretrained=True):
    """ConvNeXt-Tiny: 28.6M参数, 现代架构"""
    if pretrained:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features  # 768
    model.classifier = nn.Sequential(
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Flatten(1),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


class EnsembleModel(nn.Module):
    """模型集成: 对多个模型的logits取平均"""

    def __init__(self, model_names, pretrained=True):
        super().__init__()
        self.models = nn.ModuleList([
            get_model(name, pretrained) for name in model_names
        ])

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs, dim=0).mean(dim=0)


def freeze_backbone(model):
    """冻结backbone, 仅训练分类头"""
    for name, param in model.named_parameters():
        # 分类头参数不冻结
        if "classifier" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def unfreeze_all(model):
    """解冻全部参数"""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_parameter_groups(model, lr_backbone, lr_head):
    """获取分组参数 (不同学习率)"""
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": head_params, "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ]


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
