# 天气分类项目训练总结

## 项目概述
四类天气识别: cloudy(阴天)、rain(雨天)、sunny(晴天)、snow(下雪)
源数据: `c:/Users/19709/Desktop/睿抗/69f46e75dbb43ba9e05483c1/weather_classification/` (6类×10000张)
清洗后数据: `weather_classification/data/weather_clean/` (train 800/类, test 200/类)
剔除脏数据: 3690张, 保存于 `output/removed_images/`

## 最佳模型排行榜

| 版本 | 模型 | 配置 | Test macro F1 | cloudy F1 | 模型路径 |
|------|------|------|-------------|-----------|---------|
| Base | B0 | 224px CPU,bs=12 | 80.6% | 68.0% | output/best_model_clean.pth |
| v3 | B2 | 260px CPU,bs=12 | 81.4% | 69.8% | output/best_model_v3.pth |
| v5 | B2 | 260px GPU bs=64 AMP | 83.2% | 70.5% | output/best_model_v5_gpu.pth |
| v5+TTA | B2 | v5+TTA推理 | **83.7%** | **71.9%** | output/best_model_v5_gpu.pth |

## 最佳模型详情 (v5 GPU)

- 模型: EfficientNet-B2, 8.06M params
- 输入: 260×260px
- 分类头: 1408→256→4 (Dropout 0.4/0.25)
- GPU: NVIDIA RTX 4060 Laptop (8GB)
- 训练: batch=64, AMP, 40 epoch, EarlyStop patience=10
- 关键参数: MixUp α=0.3, CutMix α=0.2, LabelSmoothing=0.1, CosineWarmRestarts T_0=5
- 最佳epoch: 31, val F1=83.81%
- 训练耗时: ~49分钟 GPU

## v5+TTA 混淆矩阵

```
           cloudy  rain  sunny  snow
  cloudy     138    19     42      1    → 阴天误判: 晴天42, 雨天19
  rain        15   182      2      1    → 雨天误判: 阴天15
  sunny       26     4    168      2    → 晴天误判: 阴天26
  snow         5     6      7    182
```

## 关键优化经验

### 有效的
1. 数据清洗(剔除3690张脏图,重采样) → cloudy F1 +6%
2. B0→B2 backbone(5.3M→9.1M, 224→260px) → macro +0.8%, cloudy +1.8%
3. MixUp/CutMix强增强 + LabelSmoothing + CosineWarmRestarts → cloudy F1 +6%
4. GPU大batch(64) + AMP → macro +1.8%, 大幅度降低val→test过拟合
5. TTA推理(5crop+flip) → 免费 +0.5% macro, cloudy +1.4%

### 无效/失败的
- CLAHE全局应用(损害晴天)
- CBAM注意力(与EfficientNet自带SE冗余)
- 边缘通道(空间对齐bug)
- Focal Loss + MixUp冲突
- Cascade双头(过于复杂,不收敛)
- 类别加权Loss(零和博弈)

## 未来优化方向

1. 天空ROI + 阴影覆盖率 + 分块光照一致性(轻量特征,针对阴天↔晴天)
2. 地面反光 + 天空亮度 + 频域雨痕(针对阴天↔雨天)
3. 多尺度头(利用浅中深层特征)
4. 跨切分迁移训练 + Ensemble
5. 对比学习预训练

## 训练脚本对应关系

- `train_local.py` — 原始训练(旧数据)
- `train_clean.py` — Baseline(清洗后数据, CPU)
- `train_v3.py` — B2+260+EarlyStop(已被修改为v5 GPU配置)
- `train_v4.py` — B2+类别加权Loss
- `train_v6_cascade.py` — Cascade双头(失败)
- `train_v7_tta.py` — v5复现+TTA测试
- `clean_and_prepare.py` — 数据清洗脚本

## 硬件环境

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- CUDA: 13.1, PyTorch: 2.5.1+cu121
- CPU training: ~100分钟/40epoch, GPU: ~50分钟/40epoch
