"""
优化模型: EfficientNet-B0 + CBAM注意力 + HSV饱和度分支
针对阴天(cloudy)分类优化:
  - CBAM: 让模型关注判别性区域
  - HSV分支: 利用阴天低饱和度的关键特征(阴天S<30, 晴天S>80)
  - CLAHE预处理: 增强阴天图片的局部对比度(在dataset中实现)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==================== CBAM 注意力模块 ====================
class ChannelAttention(nn.Module):
    """通道注意力: 学习哪些特征通道更重要"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力: 学习图片中哪些区域更重要"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ==================== HSV 饱和度特征提取 ====================
class HSVFeatureBranch(nn.Module):
    """色彩特征分支: 轻量CNN学习饱和度/亮度等色彩分布
    阴天关键特征: 低饱和度 + 低对比度
    用浅层CNN隐式学习色彩特征, 避免显式HSV转换的复杂性
    """
    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()
        # 色彩统计 + 轻量卷积
        self.color_conv = nn.Sequential(
            # 先用1x1卷积学通道间关系(模拟色彩空间变换)
            nn.Conv2d(in_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 下采样提取色彩分布
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        feat = self.color_conv(x)
        feat = feat.squeeze(-1).squeeze(-1)
        return self.fc(feat)


# ==================== 优化后的天气分类模型 ====================
class OptimizedWeatherModel(nn.Module):
    """EfficientNet-B0 + CBAM + HSV分支
    针对阴天优化的四类天气分类器
    """
    def __init__(self, num_classes=4, dropout=0.5, hsv_dim=64):
        super().__init__()

        # 主backbone: EfficientNet-B0 (预训练)
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # 提取features部分 (去掉原始分类头)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone_feat_dim = 1280

        # CBAM注意力 (插在backbone特征之后)
        self.cbam = CBAM(backbone_feat_dim, reduction=16)

        # HSV饱和度分支
        self.hsv_branch = HSVFeatureBranch(feature_dim=hsv_dim)

        # 分类头 (RGB特征 + HSV特征融合)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_feat_dim + hsv_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # RGB主干
        rgb_feat = self.features(x)
        rgb_feat = self.avgpool(rgb_feat)
        # CBAM注意力
        rgb_feat = self.cbam(rgb_feat)
        rgb_feat = rgb_feat.mean(dim=[2, 3])  # (B, 1280)

        # HSV饱和度分支
        hsv_feat = self.hsv_branch(x)  # (B, 64)

        # 融合分类
        combined = torch.cat([rgb_feat, hsv_feat], dim=1)
        return self.classifier(combined)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
