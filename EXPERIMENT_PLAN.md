# 新闻推荐系统实验规划

## 📋 实验基本信息

**比赛名称**: 天池 - 零基础入门推荐系统 - 新闻推荐
**比赛链接**: https://tianchi.aliyun.com/competition/entrance/531842
**截止日期**: 2025年12月16日
**团队人数**: 1-2人
**Conda环境**: ds-lab2

---

## 🎯 实验目标

根据用户的历史点击日志、新闻文章特征及其向量表示，构建推荐模型，预测用户未来最可能点击的**5篇新闻文章**。

### 评价指标: MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

- 真实点击的文章在推荐列表中排名越靠前，得分越高
- 如果不在Top5中，得分为0

---

## 📊 数据说明

### 数据文件
```
Data/
├── train_click_log.csv      # 训练集用户点击日志（20万用户）
├── testA_click_log.csv       # 测试集A点击日志（5万用户）
├── articles.csv              # 文章属性信息（36万+篇）
└── articles_emb.csv          # 文章embedding向量（250维，29.5万篇）
```

### 关键数据特征

**点击日志** (`train_click_log.csv`):
- `user_id`: 用户ID（训练集：0-199999，测试集：200000-249999）
- `click_article_id`: 文章ID
- `click_timestamp`: 点击时间戳
- `click_environment`, `click_deviceGroup`, `click_os`, `click_country`, `click_region`, `click_referrer_type`: 点击环境特征

**文章信息** (`articles.csv`):
- `article_id`: 文章ID
- `category_id`: 文章类别（461个类别）
- `created_at_ts`: 文章创建时间
- `words_count`: 文章字数

**关键发现**（来自数据分析）:
1. ⚠️ **训练集和测试集用户完全不重叠** - 冷启动问题
2. 训练集用户最少点击2篇，测试集可能只有1篇
3. 用户重复点击极少（99.2%用户不重复点击）
4. 用户点击环境较稳定
5. 文章热度差异大（最热>2500次，冷门仅1-2次）

---

## 🏗️ 实验架构

采用经典的**召回-排序**两阶段架构：

```
用户历史行为
    ↓
┌───────────────────────┐
│  阶段1: 多路召回       │
│  - ItemCF召回         │
│  - Embedding召回      │
│  - YoutubeDNN召回     │
│  - UserCF召回         │
│  - 冷启动召回         │
│  召回候选集 ~150篇    │
└───────────────────────┘
    ↓
┌───────────────────────┐
│  阶段2: 特征工程       │
│  - 历史行为特征       │
│  - 用户画像特征       │
│  - 文章属性特征       │
│  - 交叉统计特征       │
└───────────────────────┘
    ↓
┌───────────────────────┐
│  阶段3: 排序模型       │
│  - LightGBM Ranker    │
│  - LightGBM Classifier│
│  - DIN深度模型        │
│  - 模型融合           │
└───────────────────────┘
    ↓
  Top 5推荐结果
```

---

## 📅 实验计划（分阶段）

### Phase 1: 环境准备与数据分析 (2-3天)

**任务清单**:
- [x] 创建conda环境 `ds-lab2`
- [ ] 安装依赖 `pip install -r requirements.txt`
- [ ] 数据加载与内存优化（`reduce_mem`函数）
- [ ] 数据探索性分析（EDA）
  - 用户点击行为分析
  - 文章属性分布分析
  - 用户-文章共现分析
  - 时间特征分析
- [ ] 可视化关键统计特征

**输出**:
- `src/data_analysis.ipynb`: 数据分析notebook
- 数据分析可视化图表
- Log: 数据加载时间、内存使用情况

---

### Phase 2: Baseline实现 (2-3天)

**任务清单**:
- [ ] 实现ItemCF相似度计算
  - 考虑时间衰减
  - 考虑位置权重
  - 考虑文章创建时间权重
- [ ] 实现ItemCF召回
  - 每用户召回Top 10文章
  - 热门文章补全策略
- [ ] 生成提交文件
- [ ] 提交baseline结果获取初始分数

**输出**:
- `src/itemcf_baseline.py`: ItemCF实现
- `user_data/itemcf_i2i_sim.pkl`: 相似度矩阵
- `prediction_result/baseline_result.csv`: 初始提交结果
- Log: 相似度计算时间、召回评估指标

---

### Phase 3: 多路召回实现 (4-5天)

**任务清单**:
- [ ] **ItemCF增强召回**
  - 结合embedding相似度
  - 优化权重参数
- [ ] **Embedding相似度召回**
  - 使用faiss加速
  - TopK=20相似文章
- [ ] **YoutubeDNN召回**
  - 构建用户序列数据
  - 训练双塔模型
  - 提取user/item embedding
  - Faiss向量检索
- [ ] **UserCF召回**（可选，内存允许）
  - 计算user相似度
  - 基于相似用户召回
- [ ] **冷启动召回**
  - 基于embedding召回候选
  - 规则过滤（主题、字数、时间）
- [ ] **多路召回融合**
  - 归一化各路得分
  - 加权融合（权重可调）
  - 每用户Top 150候选

**输出**:
- `src/recall/`:各召回模块代码
- `user_data/model_data/`: 模型文件
- `user_data/recall_results/`: 各路召回结果
- `user_data/final_recall_items_dict.pkl`: 融合召回结果
- Log: 各召回策略的hit rate、覆盖率、时间消耗

---

### Phase 4: 特征工程 (3-4天)

**任务清单**:
- [ ] **数据准备**
  - 训练/验证集划分
  - 召回结果打标签
  - 负采样（控制正负样本比例）
- [ ] **历史行为特征**
  - 候选文章与最后N次点击的相似度
  - 时间差特征
  - 字数差特征
  - 统计特征（max, min, mean, sum）
- [ ] **用户画像特征**
  - 用户活跃度（点击次数、时间间隔）
  - 设备习惯（环境、设备、OS等众数）
  - 时间习惯（点击时间统计）
  - 主题偏好（历史点击类别集合）
  - 字数偏好（历史点击平均字数）
- [ ] **文章属性特征**
  - 文章热度（点击次数、时间间隔）
  - 文章基础属性（category, words_count, created_at_ts）
- [ ] **交叉特征**
  - 候选文章主题是否在用户历史偏好中
  - Word2Vec训练文章序列embedding

**输出**:
- `src/feature_engineering.py`: 特征工程代码
- `user_data/features/`:
  - `trn_user_item_feats_df.csv`
  - `val_user_item_feats_df.csv`
  - `tst_user_item_feats_df.csv`
- `user_data/user_info.csv`: 用户画像特征
- Log: 特征构建时间、特征维度、内存使用

---

### Phase 5: 排序模型训练 (4-5天)

**任务清单**:
- [ ] **LightGBM Ranker**
  - 定义特征列
  - 构建分组信息（每用户一组）
  - 训练LGBM排序模型
  - 5折交叉验证
  - 保存验证集预测作为新特征
- [ ] **LightGBM Classifier**
  - 同样的特征
  - 二分类模型（点击/不点击）
  - 5折交叉验证
  - 输出点击概率
- [ ] **DIN模型**（深度兴趣网络）
  - 准备用户历史序列
  - 特征归一化
  - 构建attention机制
  - 模型训练
  - 5折交叉验证
- [ ] **离线评估**
  - 各模型MRR评估
  - Hit Rate @5, @10评估

**输出**:
- `src/models/`:
  - `lgb_ranker.py`
  - `lgb_classifier.py`
  - `din_model.py`
- `user_data/model_data/`:
  - 训练好的模型文件
  - 交叉验证特征
- Log:
  - 每个epoch的loss值（带时间戳）
  - 验证集评估指标
  - 训练时间

---

### Phase 6: 模型融合与优化 (2-3天)

**任务清单**:
- [ ] **简单加权融合**
  - 对3个模型的预测分数归一化
  - 调整融合权重
  - 生成融合结果
- [ ] **Stacking融合**
  - 使用5折CV生成的新特征
  - 训练简单的LR或LGBM二级模型
  - 对测试集预测
- [ ] **参数调优**
  - 召回阶段TopK调整
  - 特征选择
  - 模型超参数调优
- [ ] **结果分析**
  - 不同用户群体表现分析
  - 错误case分析

**输出**:
- `src/ensemble.py`: 融合代码
- `prediction_result/result.csv`: 最终提交结果
- Log: 不同融合策略的效果对比

---

### Phase 7: 实验报告撰写 (2-3天)

**报告结构** (参考core-requirement.md):

1. **比赛名称与队伍信息**
   - 比赛：天池 - 零基础入门推荐系统 - 新闻推荐
   - 队伍名称
   - 成员及分工

2. **问题定义**
   - 任务描述
   - 数据说明
   - 评价指标MRR解释

3. **数据分析**（Task2成果）
   - 数据统计特征
   - 可视化分析
   - 关键发现（冷启动、数据分布等）

4. **模型设计**
   - **召回阶段**（Task3）
     - 多路召回策略原理
     - ItemCF/UserCF/YoutubeDNN/Embedding召回
     - 冷启动策略
   - **特征工程**（Task4）
     - 数据预处理（原始数据→模型输入完整流程）
     - 特征分类与构建逻辑
   - **排序模型**（Task5）
     - LightGBM Ranker/Classifier原理
     - DIN模型原理与实现
     - 模型融合策略

5. **训练过程**
   - 训练集/验证集划分
   - 负采样策略
   - 超参数设置
   - 多模型对比

6. **Loss函数分析**
   - LightGBM Ranker: NDCG loss（为什么最大化NDCG能提升排序效果）
   - LightGBM Classifier: Binary Cross-Entropy（为什么最小化BCE能预测点击概率）
   - DIN: Binary Cross-Entropy（注意力机制如何帮助预测）

7. **实验结果**
   - 各阶段baseline分数
   - 模型对比结果
   - 最终排名
   - 输入输出样例展示

8. **团队分工**
   - 明确每位成员的具体工作

9. **个人总结与感悟**
   - 学到的知识
   - 遇到的困难与解决
   - 改进方向

**输出**:
- `实验报告.pdf`: 完整实验报告
- `代码.zip`: 所有源代码
- `logs/`: 带时间戳的运行日志

---

## 📝 代码组织结构

```
Lab-2/
├── Data/                          # 原始数据（不提交）
├── src/                           # 源代码
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # 数据加载工具
│   │   └── metrics.py            # 评估指标
│   ├── recall/
│   │   ├── __init__.py
│   │   ├── itemcf.py             # ItemCF召回
│   │   ├── embedding.py          # Embedding召回
│   │   ├── youtubednn.py         # YoutubeDNN召回
│   │   ├── usercf.py             # UserCF召回
│   │   └── cold_start.py         # 冷启动召回
│   ├── features/
│   │   ├── __init__.py
│   │   ├── user_features.py      # 用户特征
│   │   ├── item_features.py      # 文章特征
│   │   └── cross_features.py     # 交叉特征
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lgb_ranker.py         # LightGBM排序
│   │   ├── lgb_classifier.py     # LightGBM分类
│   │   └── din_model.py          # DIN模型
│   ├── ensemble.py                # 模型融合
│   ├── data_analysis.ipynb        # 数据分析notebook
│   └── main.py                    # 主流程
├── user_data/
│   ├── model_data/                # 模型文件
│   ├── features/                  # 特征文件
│   └── recall_results/            # 召回结果
├── prediction_result/
│   └── result.csv                 # 最终提交文件
├── logs/                          # 运行日志
├── requirements.txt               # 依赖
├── README.md                      # 运行说明
└── CLAUDE.md                      # 项目文档

```

---

## ⚠️ 重要注意事项

### 1. 冷启动问题
测试集用户从未在训练集出现，因此：
- ✅ 必须使用训练集+测试集合并数据计算文章相似度
- ✅ 不能使用仅基于用户ID的特征
- ✅ 重点关注文章相关特征和用户行为模式特征

### 2. Log文件要求
**必须记录**（带时间戳）:
```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 示例
logging.info("开始训练ItemCF模型")
logging.info(f"训练集大小: {len(train_df)}")
logging.info(f"Epoch 1, Loss: 0.1234, Time: 120s")
```

### 3. 内存优化
数据文件很大（articles_emb.csv 973MB），必须：
- 使用`reduce_mem()`函数降低DataFrame内存
- 及时删除不用的中间变量 `del var; gc.collect()`
- 大数据保存为pickle而非CSV

### 4. 提交文件格式
严格按照`results/sample_submit.csv`格式：
```csv
user_id,article_1,article_2,article_3,article_4,article_5
200000,123,456,789,234,567
200001,890,345,678,901,234
```

---

## 🎓 学习资源

### 推荐系统基础
- Datawhale推荐系统学习: https://github.com/datawhalechina/fun-rec
- YouTube DNN论文: [Deep Neural Networks for YouTube Recommendations]

### 代码参考
- 天池Baseline讨论区
- Datawhale新闻推荐Baseline (见Docs/Baseline-Task1~5/)

---

## ✅ 实验检查清单

在提交前确认：
- [ ] 代码可运行（在干净环境测试）
- [ ] Log文件包含时间戳和中间结果
- [ ] 提交文件格式正确
- [ ] 实验报告包含所有必需部分
- [ ] 团队分工明确
- [ ] 个人总结完整
- [ ] 文件命名: `姓名-学号-课程实验.zip`
- [ ] 邮件主题: `姓名-学号-课程实验`
- [ ] 发送至: data_science_2025@163.com

---

## 📧 提交信息

**截止时间**: 2025年12月16日
**提交邮箱**: data_science_2025@163.com
**文件命名**: `姓名-学号-课程实验.zip`
**邮件主题**: `姓名-学号-课程实验`

**压缩包内容**:
1. 实验报告.pdf
2. 代码文件夹（可运行）
3. logs文件夹（运行日志）
4. README.md（运行说明）

---

**祝实验顺利！🎉**
