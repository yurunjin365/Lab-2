# Loss函数详解 - 实验报告撰写参考

本文档详细说明了实验中使用的各个模型的Loss函数原理，供撰写实验报告时参考。

---

## 📌 实验报告要求

根据 `core-requirement.md`：
> **loss：报告中需要讲述loss函数的含义（即为什么最小化这个函数就可以达到分类、回归预测的目的）。**

同时，日志文件需要记录：
> **训练时的loss值，要求每一条log都记录相应的时间**

---

## 🎯 我们使用的模型及其Loss函数

### 模型1: LightGBM Ranker（排序模型）

#### 使用的Loss函数：**LambdaRank Loss**

**代码位置：** `src/models/lgb_ranker.py` 第35-36行
```python
params = {
    'objective': 'lambdarank',  # Loss函数类型
    'metric': 'ndcg',           # 评估指标
    ...
}
```

#### Loss函数原理：

**1. 什么是LambdaRank？**

LambdaRank是一种Learning to Rank（学习排序）的算法，专门用于优化排序任务。它的核心思想是：**不直接优化排序指标（如NDCG、MRR），而是优化一个可微分的损失函数，该损失函数与排序指标高度相关。**

**2. 为什么需要LambdaRank？**

在推荐系统中，我们的目标是对用户的候选文章列表进行排序，让用户真正感兴趣的文章排在最前面。传统的分类或回归loss（如Binary Cross-Entropy）只关注单个样本的预测准确性，而**不考虑样本之间的相对顺序**。

例如：
- 用户A的真实点击文章是article_123
- 候选列表：[article_123, article_456, article_789]
- 如果我们的模型预测：
  - article_123的得分：0.6
  - article_456的得分：0.8
  - article_789的得分：0.7

使用Binary Cross-Entropy只会让每个预测接近真实标签（123是1，其他是0），但不会强调"article_123必须排在最前面"。而**LambdaRank会特别惩罚"把真实点击的文章排在后面"这种错误**。

**3. LambdaRank的Loss计算过程：**

对于用户 $u$ 的候选文章列表，设：
- $s_i$ 是第 $i$ 篇文章的模型预测得分
- $y_i$ 是第 $i$ 篇文章的真实标签（1表示点击，0表示未点击）

LambdaRank的梯度计算公式：

$$\lambda_{ij} = \frac{\partial C}{\partial s_i} = -\frac{\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta NDCG_{ij}|$$

其中：
- $\lambda_{ij}$ 是文章 $i$ 相对于文章 $j$ 的梯度
- $|\Delta NDCG_{ij}|$ 是交换文章 $i$ 和 $j$ 的位置后，NDCG指标的变化量
- $\sigma$ 是sigmoid函数的参数

**关键点：**
- 如果真实点击的文章被排在未点击文章的后面，$|\Delta NDCG_{ij}|$ 会很大，梯度会很大，模型会受到强烈惩罚
- 如果排序已经很好，$|\Delta NDCG_{ij}|$ 很小，梯度很小，模型基本不需要调整

**4. 为什么最小化LambdaRank Loss可以提升排序效果？**

因为LambdaRank的梯度直接与排序指标（NDCG）挂钩：
1. 当真实点击的文章排在后面时，Loss增大，模型被惩罚
2. 模型通过梯度下降调整参数，提高真实点击文章的得分
3. 最终使得真实点击的文章排在候选列表的前面
4. 排序越好 → Loss越小 → NDCG/MRR越高

**5. Loss值在日志中的体现：**

运行代码时，日志会输出类似这样的信息：
```
2025-01-15 10:23:45 - INFO - Training until validation scores don't improve for 50 rounds
2025-01-15 10:23:50 - INFO - [50]	train's ndcg@5: 0.3421	valid's ndcg@5: 0.3156
2025-01-15 10:24:00 - INFO - [100]	train's ndcg@5: 0.4523	valid's ndcg@5: 0.4201
2025-01-15 10:24:10 - INFO - [150]	train's ndcg@5: 0.5234	valid's ndcg@5: 0.4856
```

**注意：** LightGBM的Ranker模式输出的是评估指标（NDCG），而不是直接的Loss值。这是因为LambdaRank的目标就是最大化NDCG。**NDCG越大，意味着隐式的排序Loss越小。**

---

### 模型2: LightGBM Classifier（分类模型）

#### 使用的Loss函数：**Binary Cross-Entropy Loss**

**代码位置：** `src/models/lgb_ranker.py` 第135-136行
```python
params = {
    'objective': 'binary',  # Loss函数类型（二分类）
    'metric': 'auc',        # 评估指标
    ...
}
```

#### Loss函数原理：

**1. 什么是Binary Cross-Entropy？**

Binary Cross-Entropy（二元交叉熵）是用于二分类问题的标准损失函数。在推荐系统中，我们把问题转化为：**预测用户是否会点击某篇文章（点击=1，不点击=0）。**

**2. 数学公式：**

对于单个样本 $(x, y)$，其中 $y \in \{0, 1\}$ 是真实标签，$\hat{y} = P(y=1|x)$ 是模型预测的点击概率：

$$\text{BCE Loss} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

对于整个数据集：

$$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**3. 直观理解：**

假设用户A真实点击了article_123（标签y=1），模型预测的点击概率是 $\hat{y}=0.8$：

- $\text{Loss} = -[1 \times \log(0.8) + 0 \times \log(0.2)] = -\log(0.8) \approx 0.097$

如果模型预测错了，预测概率只有 $\hat{y}=0.2$：

- $\text{Loss} = -[1 \times \log(0.2)] = -\log(0.2) \approx 0.699$

**Loss越大，说明预测越差！**

**4. 为什么最小化BCE Loss可以达到分类预测的目的？**

① **当真实标签 y=1（用户点击）时：**
   - Loss = $-\log(\hat{y})$
   - 要让Loss最小，就要让 $\hat{y}$ 尽可能接近1
   - 即模型要预测高的点击概率

② **当真实标签 y=0（用户不点击）时：**
   - Loss = $-\log(1-\hat{y})$
   - 要让Loss最小，就要让 $\hat{y}$ 尽可能接近0
   - 即模型要预测低的点击概率

③ **总体效果：**
   - 最小化BCE Loss → 让模型预测的概率分布接近真实标签分布
   - 对于点击的文章，模型输出高概率
   - 对于未点击的文章，模型输出低概率
   - 这样就能准确预测用户是否会点击某篇文章

**5. Loss值在日志中的体现：**

```
2025-01-15 10:30:15 - INFO - Training until validation scores don't improve for 50 rounds
2025-01-15 10:30:20 - INFO - [50]	train's auc: 0.7234	valid's auc: 0.7021
2025-01-15 10:30:30 - INFO - [100]	train's auc: 0.7856	valid's auc: 0.7623
2025-01-15 10:30:40 - INFO - [150]	train's auc: 0.8123	valid's auc: 0.7891
```

**注意：** LightGBM的Binary模式输出的是AUC（Area Under Curve）指标，而不是直接的BCE Loss。**AUC越大，意味着分类效果越好，BCE Loss越小。**

---

## 📊 Loss值的查看位置

### 1. 运行时的实时输出

当你运行代码时，终端会实时显示训练过程：

```bash
cd src
python main.py
```

你会看到类似这样的输出（已自动记录时间戳）：

```
2025-01-15 10:23:45 - INFO - 开始训练LightGBM Ranker模型...
2025-01-15 10:23:50 - INFO - [50]	train's ndcg@5: 0.3421	valid's ndcg@5: 0.3156
2025-01-15 10:24:00 - INFO - [100]	train's ndcg@5: 0.4523	valid's ndcg@5: 0.4201
2025-01-15 10:24:10 - INFO - [150]	train's ndcg@5: 0.5234	valid's ndcg@5: 0.4856
```

### 2. 日志文件

所有输出都会自动保存到 `logs/` 目录：

```bash
ls logs/
# 输出: main_20250115_102345.log

cat logs/main_20250115_102345.log
```

日志文件包含：
- ✅ 每步操作的时间戳
- ✅ 模型训练的loss/metric变化
- ✅ 数据处理的统计信息

### 3. 如何在实验报告中使用

**示例写法：**

> #### 5.1 LightGBM Ranker训练过程
>
> 我们使用LightGBM的排序模型（Ranker），其损失函数为LambdaRank。该损失函数专门为排序任务设计，通过优化NDCG指标来提升排序效果。
>
> **Loss函数原理：**
>
> LambdaRank不直接优化排序指标，而是计算每对文章之间的梯度，梯度大小与交换两篇文章位置后NDCG的变化量成正比。当真实点击的文章被排在未点击文章后面时，模型会受到强烈惩罚，从而学习到正确的排序。
>
> **训练过程：**
>
> 根据日志记录（logs/main_20250115_102345.log），模型训练过程如下：
>
> | 迭代轮数 | 训练集NDCG@5 | 验证集NDCG@5 | 时间 |
> |---------|-------------|-------------|------|
> | 50      | 0.3421      | 0.3156      | 10:23:50 |
> | 100     | 0.4523      | 0.4201      | 10:24:00 |
> | 150     | 0.5234      | 0.4856      | 10:24:10 |
>
> 模型在第150轮达到最佳效果，验证集NDCG@5为0.4856。随着训练进行，NDCG指标不断上升，说明模型的排序能力在增强，隐式的排序Loss在降低。

---

## 🎓 实验报告撰写建议

### 必需包含的内容：

1. **Loss函数的数学定义**
   - LambdaRank的梯度公式
   - Binary Cross-Entropy的公式

2. **Loss函数的物理意义**
   - 为什么这个loss能解决我们的问题
   - 最小化loss如何帮助模型学习

3. **训练过程的Loss变化**
   - 贴出日志中的训练记录（带时间戳）
   - 可以绘制Loss/Metric变化曲线图

4. **多个模型的Loss对比**
   - LightGBM Ranker vs Classifier
   - 解释为什么选择这两种loss

### 可选的进阶内容：

- Loss函数与评估指标（MRR/NDCG）的关系
- 不同loss函数的优缺点对比
- 如果使用了模型融合，如何综合多个模型的预测结果

---

## 📝 快速参考表

| 模型 | Loss函数 | 代码位置 | 日志指标 | 报告关键词 |
|------|---------|---------|---------|-----------|
| LightGBM Ranker | LambdaRank | lgb_ranker.py:35 | NDCG@5 | 排序loss、Learning to Rank |
| LightGBM Classifier | Binary Cross-Entropy | lgb_ranker.py:135 | AUC | 二分类loss、点击率预测 |

---

**总结：实验报告需要详细解释每个模型的loss函数原理，并从日志文件中提取训练过程的metric变化来证明模型在不断学习和优化。**
