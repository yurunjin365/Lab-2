# 新闻推荐系统实验

## 项目简介

本项目是针对天池"零基础入门推荐系统 - 新闻推荐"比赛的完整实现，采用经典的**召回-排序**两阶段架构，实现新闻文章的个性化推荐。

**比赛链接**: https://tianchi.aliyun.com/competition/entrance/531842

## 环境准备

### 1. 创建Conda环境

```bash
conda create -n ds-lab2 python=3.8
conda activate ds-lab2
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将比赛数据下载并放置在`Data/`目录下：

```
Data/
├── train_click_log.csv
├── testA_click_log.csv
├── articles.csv
└── articles_emb.csv
```

## 快速开始

### 方式1: 运行完整流程

```bash
python src/main.py
```

这将依次执行：
1. 数据分析
2. 多路召回
3. 特征工程
4. 模型训练
5. 模型融合
6. 生成提交文件

### 方式2: 分步执行

```bash
# 1. 数据分析
jupyter notebook src/data_analysis.ipynb

# 2. ItemCF召回（baseline）
python src/recall/itemcf.py

# 3. 多路召回
python src/recall/embedding.py
python src/recall/youtubednn.py

# 4. 特征工程
python src/features/feature_engineering.py

# 5. 模型训练
python src/models/lgb_ranker.py
python src/models/lgb_classifier.py
python src/models/din_model.py

# 6. 模型融合
python src/ensemble.py
```

## 项目结构

```
Lab-2/
├── Data/                      # 原始数据
├── src/                       # 源代码
│   ├── utils/                # 工具函数
│   ├── recall/               # 召回模块
│   ├── features/             # 特征工程
│   ├── models/               # 排序模型
│   └── main.py               # 主流程
├── user_data/                # 中间数据
│   ├── model_data/           # 模型文件
│   ├── features/             # 特征文件
│   └── recall_results/       # 召回结果
├── prediction_result/        # 预测结果
│   └── result.csv            # 最终提交文件
├── logs/                     # 运行日志
├── requirements.txt          # Python依赖
├── CLAUDE.md                 # 项目文档
├── EXPERIMENT_PLAN.md        # 实验规划
└── README.md                 # 本文件
```

## 核心流程

### 1. 多路召回阶段

实现5种召回策略：
- **ItemCF召回**: 基于物品协同过滤，考虑时间衰减和位置权重
- **Embedding召回**: 使用文章embedding + faiss加速检索
- **YoutubeDNN召回**: 双塔模型学习用户和文章表示
- **UserCF召回**: 基于用户协同过滤（可选）
- **冷启动召回**: 针对未出现文章的规则召回

每路召回约10-20篇候选，融合后每用户保留Top 150候选。

### 2. 特征工程

构建三类特征：
- **用户画像特征**: 活跃度、设备习惯、时间习惯、主题偏好、字数偏好
- **文章属性特征**: 热度、类别、字数、创建时间
- **交叉特征**: 候选文章与用户历史行为的相似度、时间差、字数差等

### 3. 排序模型

训练3个排序模型：
- **LightGBM Ranker**: 学习排序任务
- **LightGBM Classifier**: 点击率预测
- **DIN模型**: 深度兴趣网络，使用attention机制

### 4. 模型融合

- 简单加权融合
- Stacking融合（LR或LGBM二级模型）

## 输出文件

### 最终提交文件

`prediction_result/result.csv` 格式：

```csv
user_id,article_1,article_2,article_3,article_4,article_5
200000,123,456,789,234,567
200001,890,345,678,901,234
...
```

### 日志文件

所有运行日志保存在`logs/`目录，包含时间戳和中间结果：

```
2024-12-16 10:15:23 - INFO - 开始ItemCF相似度计算
2024-12-16 10:18:45 - INFO - 相似度矩阵计算完成，耗时202秒
2024-12-16 10:20:00 - INFO - Epoch: 1, Step: 100, Loss: 0.1234
...
```

## 评价指标

**MRR (Mean Reciprocal Rank)**:

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

- 真实点击的文章在推荐Top 5中的位置越靠前，分数越高
- 不在Top 5中则得分为0

## 关键技巧

### 1. 冷启动处理

⚠️ **重要**: 测试集用户完全不在训练集中！

解决方案：
- 使用 `train + test` 合并数据计算文章相似度
- 重点使用文章相关特征，而非用户ID特征
- 实现专门的冷启动召回策略

### 2. 内存优化

数据文件较大，使用以下技术：
```python
# 内存压缩
df = reduce_mem(df)

# 及时清理
del large_var
gc.collect()

# 使用pickle而非CSV
import pickle
pickle.dump(data, open('file.pkl', 'wb'))
```

### 3. 负采样

召回候选中正样本极少，需要负采样：
```python
# 保证每个用户和每篇文章都出现
neg_sample_by_user = neg_data.groupby('user_id').sample(...)
neg_sample_by_item = neg_data.groupby('item_id').sample(...)
```

## 常见问题

### Q1: 内存不足怎么办？
- 使用`reduce_mem()`压缩DataFrame
- 分批处理大数据
- 删除不用的中间变量

### Q2: YoutubeDNN训练太慢？
- 减少embedding维度（16维即可）
- 减少序列长度（max_len=30）
- 使用采样数据调试

### Q3: 如何验证模型效果？
- 从训练集抽取验证集
- 计算MRR指标
- 5折交叉验证

## 参考资料

- [Datawhale推荐系统学习](https://github.com/datawhalechina/fun-rec)
- [天池Baseline讨论区](https://tianchi.aliyun.com/competition/entrance/531842/forum)
- [YouTube DNN论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

## 联系方式

- 课程邮箱: data_science_2025@163.com
- 比赛讨论: 天池比赛论坛

## License

本项目仅用于课程学习使用。
