# Weather Classification

四类天气识别：cloudy(阴天) / rain(雨天) / sunny(晴天) / snow(雪天)

## Best Model

**ConvNeXt-Tiny + TTA + 2-Model Ensemble: 88.46% Macro F1**

| Model | Backbone | Macro F1 | Notes |
|-------|----------|---------|-------|
| v10 | ConvNeXt-Tiny | 87.91% | Single model s456 |
| v10 Ensemble | ConvNeXt-Tiny x2 | 88.46% | s42 + s456 + threshold |

## Quick Start

```python
from handler import handle
result = handle("image.jpg", model_path="output/best_model_v10_convnext_s456.pth")
print(result["prediction_cn"])  # 阴天/雨天/晴天/雪天
```

## Training

```bash
python train_v10_convnext.py
```
