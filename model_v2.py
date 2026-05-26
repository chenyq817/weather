"""
EfficientNet-B0 修改第一层接受 4 通道输入 (RGB + 边缘密度图)
边缘密度: 阴天低(漫射光无锐利阴影), 晴天高(方向性阴影)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_edge_model(num_classes=4, dropout=0.5, pretrained=True):
    """构建接受4通道(RGB+Edge)输入的EfficientNet-B0"""
    # 加载预训练模型
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # 获取原始第一层卷积
    old_conv = model.features[0][0]  # Conv2d(3, 32, 3, stride=2)
    out_channels = old_conv.out_channels

    # 创建新的4通道卷积
    new_conv = nn.Conv2d(
        4, out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # 初始化: 前3通道复制预训练权重, 第4通道(边缘)初始化为前3通道均值
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight.clone()
        new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True) * 0.1

    model.features[0][0] = new_conv

    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=dropout * 0.6),
        nn.Linear(256, num_classes),
    )

    return model


class FocalLoss(nn.Module):
    """Focal Loss: 自动降低易分样本权重, 聚焦困难样本
    gamma=2: 标准设置, 对高置信度样本大幅降权
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重, 可选
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none',
                                               weight=self.alpha)
        p = torch.exp(-ce_loss)  # 预测概率
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
