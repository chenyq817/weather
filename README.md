# 天气图片分类 - 智海算法调优

四类天气识别：cloudy(阴天) / rain(雨天) / shine(晴天) / sunrise(日出)

数据集: Kaggle [Multi-class Weather Dataset (MWD)](https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset)

## Kaggle MWD → Mo平台 部署步骤

### 1. 下载数据集

访问 https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset
点击 Download 下载 archive.zip（约30MB，1125张图片）

### 2. 上传到Mo平台

登录Mo平台 → 新建项目 → 上传文件：
- `coding_here.ipynb`
- `archive.zip`（Kaggle下载的数据集）

### 3. 运行Notebook

在Mo平台打开 `coding_here.ipynb`，按顺序运行每个Cell：

| Cell | 作用 |
|------|------|
| Step 0 | 安装依赖 |
| 解压Cell | 解压 archive.zip |
| Step 1 | 配置参数（GPU环境自动检测） |
| Step 2 | 数据增强管线 |
| Step 3 | 自动查找并加载数据 |
| Step 4-7 | 构建模型 + 训练函数 |
| Step 8 | 开始训练（GPU建议60轮） |
| Step 9-10 | 推理测试 + Handle函数 |

### 4. 部署应用

- 左侧栏 → 部署图标
- 选中 Step 10 的 handle 函数 → 插入
- 勾选依赖Cell + best_model.pth
- 发布正式版本

### 5. 测试

部署后在详情页点"测试项目"，上传天气图片验证分类结果。

## 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 本地训练（需要先下载Kaggle数据集到 data/ 目录）
python train_local.py
```
