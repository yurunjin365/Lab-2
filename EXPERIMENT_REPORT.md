# 新闻推荐系统实验报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **比赛名称** | 天池 - 零基础入门推荐系统 - 新闻推荐 |
| **队伍名** | [请填写] |
| **成员** | [请填写] |
| **比赛排名** | [待提交后填写] |

---

## 一、问题定义

### 1.1 任务描述

本赛题以**预测用户未来点击新闻文章**为任务。选手需要根据用户的历史点击日志、新闻文章特征及其向量表示，构建一个推荐模型，预测用户在未来最可能点击的新闻文章。

### 1.2 数据说明

| 数据文件 | 描述 | 规模 |
|---------|------|------|
| `train_click_log.csv` | 训练集用户点击日志 | 20万用户，约111万条记录 |
| `testA_click_log.csv` | 测试集用户点击日志 | 5万用户，约52万条记录 |
| `articles.csv` | 文章属性信息 | 36万+篇文章 |
| `articles_emb.csv` | 文章Embedding向量 | 250维向量表示 |

**数据字段说明**：

```
用户点击日志：
- user_id: 用户ID
- click_article_id: 点击的文章ID
- click_timestamp: 点击时间戳
- click_environment: 点击环境
- click_deviceGroup: 设备类型
- click_os: 操作系统
- click_country: 国家
- click_region: 地区
- click_referrer_type: 来源类型

文章信息：
- article_id: 文章ID
- category_id: 类别ID
- created_at_ts: 创建时间戳
- words_count: 字数
```

### 1.3 评估指标

本赛题采用 **MRR (Mean Reciprocal Rank)** 进行评价：

$$MRR = \frac{1}{|U|} \sum_{u \in U} \frac{1}{rank_u}$$

其中：
- $|U|$ 是测试集用户总数
- $rank_u$ 是用户 $u$ 真实点击文章在推荐列表中的排名位置

**直观理解**：如果真实点击的文章排在推荐列表第1位得1分，第2位得0.5分，第3位得0.33分...MRR越高越好。

---

## 二、数据分析

### 2.1 数据概览

通过运行 `data_analysis.py`，我们对数据进行了全面分析：

```
训练集统计：
- 用户数：200,000
- 点击记录数：1,112,623
- 文章数：约36,000篇
- 平均每用户点击：5.56篇

测试集统计：
- 用户数：50,000
- 点击记录数：518,010
- 平均每用户点击：10.36篇
```

### 2.2 用户行为分析

1. **点击次数分布**：大部分用户点击次数在2-10次之间，呈长尾分布
2. **时间分布**：点击行为有明显的时间规律（早晚高峰）
3. **设备分布**：移动端占主导

### 2.3 文章特征分析

1. **类别分布**：文章类别分布不均匀，存在热门类别
2. **字数分布**：文章字数集中在200-1000字
3. **热门文章**：少数文章获得大量点击（马太效应）

### 2.4 数据处理流程

**原始数据 → 模型输入的转换过程**：

```
原始点击日志（一行）:
user_id=12345, click_article_id=67890, click_timestamp=1507029570, ...

                    ↓ 转换过程

1. 获取用户历史点击序列：[article_1, article_2, ..., article_n]
2. 召回候选文章：通过ItemCF/Embedding找到相似文章
3. 构建样本特征：
   - 用户特征：点击次数、活跃度、偏好类别...
   - 文章特征：类别、字数、热度、新鲜度...
   - 交叉特征：用户-文章相似度、历史交互...
4. 生成标签：候选文章是否被真实点击（0/1）

模型输入（一行）:
[user_click_count, item_hot_score, sim_score, category_match, ...] → label(0/1)
```

---

## 三、解题思路与模型设计

### 3.1 整体架构

采用业界主流的 **"召回-排序"** 两阶段架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     全部文章池（36万+）                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  【召回阶段】快速筛选，从36万缩小到150篇候选                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   ItemCF    │  │  Embedding  │  │   热门召回   │         │
│  │   召回      │  │    召回     │  │  (冷启动)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓                ↓                ↓                 │
│                  加权融合 (0.6:0.4)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  【排序阶段】精准排序，选出Top 5推荐                           │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ LightGBM Ranker │    │LightGBM Classifier│               │
│  │  (排序模型)      │    │  (点击率预测)     │               │
│  └─────────────────┘    └─────────────────┘                │
│            ↓                     ↓                          │
│                  模型融合 (0.6:0.4)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Top 5 推荐结果                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 召回策略

#### 3.2.1 ItemCF（基于物品的协同过滤）

**原理**：利用用户的历史点击行为，找到与历史点击文章相似的文章作为候选。

**相似度计算公式**：

$$sim(i, j) = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \times |U_j|}} \times w_{time} \times w_{pos} \times w_{fresh}$$

其中：
- $|U_i \cap U_j|$：同时点击文章i和j的用户数
- $w_{time}$：时间衰减权重（点击间隔越近权重越高）
- $w_{pos}$：位置权重（点击顺序靠前权重越高）
- $w_{fresh}$：文章新鲜度权重

**实现代码参考**：`src/recall/itemcf.py`

#### 3.2.2 Embedding召回

**原理**：利用文章的预训练Embedding向量，通过向量相似度找到相似文章。

**相似度计算**：使用Faiss库进行高效的向量检索

$$sim(i, j) = \frac{emb_i \cdot emb_j}{||emb_i|| \times ||emb_j||}$$

**实现代码参考**：`src/recall/embedding.py`

### 3.3 排序模型

#### 3.3.1 LightGBM Ranker

**模型选择依据**：
- GBDT类模型在推荐排序任务中效果稳定
- LightGBM训练速度快，支持大规模数据
- 原生支持Learning to Rank任务

**模型原理**：
- 目标函数：LambdaRank
- 优化目标：最大化NDCG（Normalized Discounted Cumulative Gain）

#### 3.3.2 LightGBM Classifier

**模型原理**：
- 目标函数：Binary Cross-Entropy
- 将排序问题转化为点击率预测的二分类问题

### 3.4 特征工程

| 特征类别 | 特征名称 | 描述 |
|---------|---------|------|
| 用户特征 | user_click_count | 用户历史点击总数 |
| | user_active_days | 用户活跃天数 |
| 文章特征 | item_click_count | 文章被点击总数 |
| | item_words_count | 文章字数 |
| | item_created_time | 文章创建时间 |
| 交叉特征 | itemcf_sim_score | ItemCF相似度分数 |
| | emb_sim_score | Embedding相似度分数 |
| | category_match | 类别匹配度 |

---

## 四、Loss函数说明

### 4.1 LightGBM Ranker - LambdaRank Loss

**目标**：学习文档的相对排序，使相关文档排在不相关文档前面。

**原理**：
LambdaRank是一种Pairwise的排序学习方法，它不直接优化排序指标（如NDCG），而是通过梯度来隐式优化。

**Lambda梯度定义**：

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \times |\Delta NDCG_{ij}|$$

其中：
- $s_i, s_j$ 是文档i和j的预测分数
- $|\Delta NDCG_{ij}|$ 是交换i和j位置后NDCG的变化量

**为什么最小化这个函数能达到排序目的**：
- Lambda梯度考虑了排序位置的重要性
- 高位置的错误排序会产生更大的梯度
- 模型会优先纠正对排序指标影响大的样本对

### 4.2 LightGBM Classifier - Binary Cross-Entropy Loss

**目标**：预测用户点击文章的概率。

**公式**：

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

其中：
- $y_i$ 是真实标签（1=点击，0=未点击）
- $p_i$ 是模型预测的点击概率

**为什么最小化这个函数能达到分类目的**：
- 当真实标签为1时，$-\log(p_i)$ 要求 $p_i$ 尽可能接近1
- 当真实标签为0时，$-\log(1-p_i)$ 要求 $p_i$ 尽可能接近0
- 最小化交叉熵等价于最大化预测概率与真实标签的一致性

---

## 五、训练过程

### 5.1 数据划分

```
训练集(20万用户)
    ├── 训练历史：每个用户除最后一次点击外的所有点击
    └── 验证集：每个用户最后一次点击（用于离线评估）

测试集(5万用户)
    └── 用于最终预测提交
```

### 5.2 负采样策略

由于正样本（真实点击）远少于负样本（召回但未点击），采用负采样平衡数据：

- 正负样本比例：1:5
- 采样方法：智能采样（保证每个用户和文章都有负样本）

### 5.3 超参数设置

**LightGBM Ranker**：
```python
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}
```

**LightGBM Classifier**：
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}
```

### 5.4 训练日志示例

```
[待模型训练完成后填入实际日志]

示例格式：
2025-12-16 22:30:15 - INFO - 开始训练LightGBM Ranker模型...
[50]  train's ndcg@5: 0.xxxx
[100] train's ndcg@5: 0.xxxx
...
Early stopping at round xxx
```

### 5.5 模型比较

| 模型 | 离线指标 | 线上MRR | 备注 |
|------|---------|---------|------|
| ItemCF Baseline | - | [待填写] | 纯召回，无排序 |
| LightGBM Ranker | NDCG@5: [待填写] | [待填写] | 排序模型 |
| LightGBM Classifier | AUC: [待填写] | [待填写] | 分类模型 |
| 模型融合 | - | [待填写] | 最终提交 |

---

## 六、预测结果展示

### 6.1 输入输出样例

**输入**（用户历史点击）：
```
user_id: 249999
历史点击: [article_123, article_456, article_789, ...]
```

**输出**（Top 5推荐）：
```
user_id,article_1,article_2,article_3,article_4,article_5
249999,95716,234698,158794,288320,233478
```

### 6.2 提交文件格式

```csv
user_id,article_1,article_2,article_3,article_4,article_5
249999,95716,234698,158794,288320,233478
249998,16129,300470,276970,95972,233420
...
```

### 6.3 最终结果

| 指标 | 数值 |
|------|------|
| 提交文件 | `prediction_result/final_result.csv` |
| 测试集用户数 | 50,000 |
| 线上MRR得分 | [待提交后填写] |
| 比赛排名 | [待提交后填写] |

---

## 七、团队分工

| 成员 | 分工内容 |
|------|---------|
| [成员1] | [请填写具体分工] |
| [成员2] | [请填写具体分工] |

---

## 八、个人总结与感悟

### [成员姓名]

[请填写个人总结，包括：]
- 在项目中的具体贡献
- 学习到的知识和技能
- 遇到的困难和解决方法
- 对推荐系统的理解和感悟

---

## 附录

### A. 代码结构

```
Lab-2/
├── Data/                      # 数据目录
│   ├── train_click_log.csv
│   ├── testA_click_log.csv
│   ├── articles.csv
│   └── articles_emb.csv
├── src/                       # 源代码
│   ├── main.py               # 主流程
│   ├── baseline.py           # 基线模型
│   ├── data_analysis.py      # 数据分析
│   ├── recall/               # 召回模块
│   │   ├── itemcf.py
│   │   └── embedding.py
│   ├── features/             # 特征工程
│   │   └── feature_engineering.py
│   ├── models/               # 模型
│   │   └── lgb_ranker.py
│   └── utils/                # 工具函数
├── logs/                      # 运行日志
├── prediction_result/         # 预测结果
├── user_data/                 # 中间数据
└── requirements.txt           # 依赖
```

### B. 运行说明

```bash
# 1. 创建环境
conda create -n ds-lab2 python=3.11
conda activate ds-lab2

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行完整流程
python src/main.py

# 4. 或运行基线模型
python src/baseline.py
```

### C. 参考资料

1. 天池竞赛官方文档
2. LightGBM官方文档：https://lightgbm.readthedocs.io/
3. 推荐系统实践相关论文和博客

