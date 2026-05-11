# 天气图片分类 - 智海算法调优

四类天气识别：晴天(sunny) / 雨天(rainy) / 阴天(cloudy) / 雪天(snowy)

## 智海Mo平台部署指南

### 第一步：新建项目

1. 登录 [智海Mo平台](https://mo.zju.edu.cn/) 工作台
2. 点击 **新建项目**，项目名称填写 `天气图片分类`
3. 进入开发页面，你会看到一个 `coding_here.ipynb` 文件

### 第二步：导入代码到Notebook

有两种方式：

**方式A（推荐）：上传完整Notebook**
- 在左侧文件栏点击上传按钮，上传本项目的 `coding_here.ipynb` 文件
- 打开该notebook

**方式B：逐Cell复制代码**
- 打开 `coding_here.ipynb`
- 将本项目的 `coding_here.ipynb` 内容复制到Mo平台的notebook中

### 第三步：插入数据集模块

1. 在Notebook中选中一个空Cell
2. 点击左侧 **模块图标**（方块形状）
3. 搜索框输入 `weather` 或 `天气`，找到天气数据集模块
4. 点击 **插入模块** 按钮
5. 如果平台没有天气数据集模块，可以从以下来源获取：
   - **Multi-class Weather Dataset (MWD)**: https://data.mendeley.com/datasets/4drtyfjtfy/1
   - 上传到平台的 `./data/` 目录，按类别文件夹组织

### 第四步：按顺序运行Cells

1. 点击 **Step 0** Cell：安装依赖
2. 点击 **Step 1** Cell：配置参数（可根据需要调整 `NUM_EPOCHS` 等）
3. 点击 **Step 2** Cell：数据增强管线
4. 点击 **Step 3** Cell：加载数据集（**修改 `DATA_PATH` 为你的数据集路径**）
5. 点击 **Step 4** Cell：创建DataLoader
6. 点击 **Step 5** Cell：构建模型
7. 点击 **Step 6** Cell：MixUp函数
8. 点击 **Step 7** Cell：训练函数
9. 点击 **Step 8** Cell：**开始训练**（这是核心步骤）
10. 点击 **Step 9** Cell：推理测试
11. 点击 **Step 10** Cell：Handle函数定义

> **速度提示**：训练时间较长时，可以创建一个Job并使用GPU加速。在左侧栏找到 **Job** 图标创建。

### 第五步：部署应用

训练完成并获得满意的模型后：

#### 5.1 打开部署页面
点击左侧栏的 **部署图标**（火箭形状）

#### 5.2 插入Handle函数并配置参数
1. **选中** Step 10 中 `handle` 函数所在的Cell
2. 点击部署页面 **第一步的"插入"按钮**，系统会自动识别handle函数
3. handle函数的输入参数：
   - `image_input` (str, 必填): 图片路径或Base64编码
   - `model_path` (str, 选填): 模型路径，默认 `best_model.pth`
4. handle函数的输出：
   - `prediction`: 天气类别
   - `prediction_cn`: 中文天气类别
   - `confidence`: 置信度
   - `probabilities`: 各类别概率
   - `inference_time_ms`: 推理耗时

#### 5.3 准备部署文件
1. 点击第二步的 **"开始"按钮**
2. 勾选以下文件：
   - **handle函数所在的Cell**（必须）
   - `coding_here.ipynb` 中的 **模型定义、预处理等依赖Cell**（必须）
   - `best_model.pth` 模型文件（必须）
3. 预览生成的代码，确认无误
4. 系统会自动识别输入输出参数，生成 `app_spec.yml` 配置文件
5. 也可以用本项目的 [app_spec.yml](app_spec.yml) 和 [handler.py](handler.py) 直接替换

#### 5.4 部署
1. 点击第三步的 **"部署"按钮**
2. 选择发布版本类型：
   - **开发版本**：仅自己可用（适合调试）
   - **正式版本**：所有人可见（比赛提交选这个）
3. 点击 **"完成"**

### 第六步：测试已部署的应用

1. 在部署栏中点击 **测试项目**
2. 输入一张天气图片的URL或上传图片
3. 点击运行，查看返回的分类结果

### 关键调优参数

| 参数 | 位置 | 建议值 | 说明 |
|------|------|--------|------|
| `NUM_EPOCHS` | Step 1 | 30~60 | 训练轮数，越大越准但慢 |
| `BATCH_SIZE` | Step 1 | 32~64 | GPU显存不够就改小 |
| `LEARNING_RATE` | Step 1 | 1e-3 | 学习率 |
| `LABEL_SMOOTHING` | Step 1 | 0.1 | 标签平滑，防过拟合 |

### 常见问题

**Q: 找不到数据集模块？**
A: 从 Mendeley 下载 MWD 数据集上传到 `./data/train/` 按类别分文件夹。

**Q: 训练很慢？**
A: 创建Job使用GPU加速：左侧栏 → Job图标 → 新建Job → 选择GPU。

**Q: 部署后推理报错？**
A: 检查 `best_model.pth` 是否包含在部署文件中，以及 `torch`、`torchvision` 依赖是否正确。

**Q: F1分数不够高？**
A: 增大 `NUM_EPOCHS=60`，开启 `TTA` 测试时增强，或换用 `densenet121` 模型。
