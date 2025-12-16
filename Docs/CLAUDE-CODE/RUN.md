# 新闻推荐系统 - 运行说明

## 📋 前置准备

### 1. 环境安装

确保你在`ds-lab2` conda环境中，并安装所有依赖：

```bash
conda activate ds-lab2
pip install -r requirements.txt
```

### 2. 数据准备

确保`Data/`目录下有以下文件：
- `train_click_log.csv` - 训练集点击日志
- `testA_click_log.csv` - 测试集点击日志
- `articles.csv` - 文章信息
- `articles_emb.csv` - 文章embedding向量

---

## 🚀 快速开始

### 方式1：Baseline快速测试（推荐先运行）

只使用ItemCF召回，不需要特征工程和排序模型，可以快速得到初始结果：

```bash
cd src
conda run -n ds-lab2 python baseline.py
```

**预计耗时：** 10-20分钟（取决于机器性能）

**输出文件：**
- `prediction_result/baseline_result.csv` - 提交文件
- `user_data/itemcf_i2i_sim.pkl` - ItemCF相似度矩阵（会被保存复用）
- `logs/baseline_YYYYMMDD_HHMMSS.log` - 运行日志

---

### 方式2：完整Pipeline

完整的召回-排序-融合流程：

```bash
cd src
conda run -n ds-lab2 python main.py
```

**预计耗时：** 1-2小时（取决于机器性能和数据量）

**流程说明：**
1. 数据加载与内存优化
2. ItemCF + Embedding双路召回
3. 召回结果融合（Top 150候选）
4. 特征工程（用户特征、文章特征、交叉特征）
5. LightGBM Ranker + Classifier训练
6. 模型融合（0.6 Ranker + 0.4 Classifier）
7. 生成提交文件

**输出文件：**
- `prediction_result/final_result.csv` - 最终提交文件
- `user_data/itemcf_i2i_sim.pkl` - ItemCF相似度矩阵
- `user_data/emb_i2i_sim.pkl` - Embedding相似度矩阵
- `user_data/recall_results/` - 召回结果
- `user_data/features/` - 特征文件
- `user_data/model_data/` - 训练好的模型
- `logs/main_YYYYMMDD_HHMMSS.log` - 详细运行日志

---

### 方式3：数据分析

在运行模型之前，可以先运行数据分析了解数据特征：

```bash
cd src
conda run -n ds-lab2 python data_analysis.py
```

**输出：**
- 用户点击行为分析
- 文章属性统计
- 用户-文章交互分析
- 时间特征分析
- 训练集-测试集重叠分析（⚠️ 会发现用户完全不重叠）

---

## 📂 项目结构

```
Lab-2/
├── Data/                      # 原始数据（运行前需准备）
├── src/                       # 源代码
│   ├── utils/                # 工具函数
│   │   ├── data_loader.py    # 数据加载、内存优化
│   │   └── metrics.py        # 评估指标（MRR）
│   ├── recall/               # 召回模块
│   │   ├── itemcf.py         # ItemCF召回
│   │   └── embedding.py      # Embedding召回
│   ├── features/             # 特征工程
│   │   └── feature_engineering.py
│   ├── models/               # 排序模型
│   │   └── lgb_ranker.py     # LightGBM Ranker/Classifier
│   ├── baseline.py           # ⭐ Baseline快速测试脚本
│   ├── main.py               # ⭐ 完整流程主脚本
│   └── data_analysis.py      # 数据分析脚本
├── user_data/                # 中间数据（运行时自动生成）
│   ├── model_data/           # 模型文件
│   ├── features/             # 特征文件
│   └── recall_results/       # 召回结果
├── prediction_result/        # 预测结果（运行时自动生成）
│   ├── baseline_result.csv   # Baseline提交文件
│   └── final_result.csv      # 最终提交文件
├── logs/                     # 运行日志（运行时自动生成）
└── requirements.txt          # Python依赖
```

---

## 🔍 提交文件格式

生成的`result.csv`格式如下：

```csv
user_id,article_1,article_2,article_3,article_4,article_5
200000,123456,234567,345678,456789,567890
200001,111111,222222,333333,444444,555555
...
```

- 每行代表一个用户
- 推荐5篇文章，按相关性降序排列
- 第1篇文章最相关，第5篇最不相关
- 评估指标MRR会根据真实点击在Top5中的位置计算得分

---

## 🛠️ 常见问题

### Q1: 内存不足怎么办？

**解决方案：**
1. 使用`reduce_mem()`函数已经自动优化了内存
2. 如果还是不够，可以：
   - 减少召回候选数量（修改`item_num`参数）
   - 减少特征数量
   - 分批处理数据

### Q2: 运行时间太长怎么办？

**优化建议：**
1. 先运行`baseline.py`快速测试
2. 检查是否有已保存的中间结果可以复用（`.pkl`文件）
3. 调整模型参数：
   - 减少`num_boost_round`（LightGBM迭代次数）
   - 减少召回路数（可以只用ItemCF）

### Q3: faiss安装失败怎么办？

代码已经做了兼容处理：
- 如果faiss可用，使用faiss加速向量检索
- 如果faiss不可用，自动降级为numpy计算（稍慢但可用）

### Q4: 如何查看运行日志？

所有日志都保存在`logs/`目录下，文件名包含时间戳：
```bash
# 查看最新的日志
ls -lt logs/
tail -f logs/main_*.log
```

日志包含：
- ✅ 每个步骤的开始/结束时间
- ✅ 数据统计信息
- ✅ 模型训练进度
- ✅ 中间结果评估

### Q5: 如何调整模型参数？

修改`src/main.py`中的相关参数：

```python
# 召回阶段
top_k=10,          # 每个历史物品召回的相似物品数
item_num=100,      # 每路召回的候选数
topk=150,          # 融合后的总候选数

# 召回融合权重
weights=[0.6, 0.4],  # [ItemCF权重, Embedding权重]

# 排序模型融合
0.6 * ranker + 0.4 * classifier  # 可调整权重
```

---

## 📊 评估指标说明

**MRR (Mean Reciprocal Rank)**

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

- 真实点击在推荐Top5中排第1位：得分 1.0
- 真实点击在推荐Top5中排第2位：得分 0.5
- 真实点击在推荐Top5中排第3位：得分 0.33
- 真实点击在推荐Top5中排第4位：得分 0.25
- 真实点击在推荐Top5中排第5位：得分 0.2
- 真实点击不在Top5中：得分 0

**目标：** 让用户真实点击的文章排在推荐列表越靠前越好！

---

## 💡 代码特点

### 1. 完善的日志系统
- ✅ 所有操作都有时间戳记录
- ✅ 记录中间结果和统计信息
- ✅ 满足实验报告要求

### 2. 内存优化
- ✅ 自动压缩DataFrame数据类型
- ✅ 使用pickle保存大文件
- ✅ 及时清理中间变量

### 3. 冷启动处理
- ✅ 使用训练集+测试集合并数据计算相似度
- ✅ 热门文章填充策略
- ✅ 重点关注文章特征而非用户ID特征

### 4. 模块化设计
- ✅ 清晰的代码结构
- ✅ 可复用的工具函数
- ✅ 易于扩展和调试

---

## 📝 下一步

1. **运行baseline获得初始结果**
   ```bash
   cd src
   conda run -n ds-lab2 python baseline.py
   ```

2. **运行完整流程提升效果**
   ```bash
   cd src
   conda run -n ds-lab2 python main.py
   ```

3. **查看日志分析结果**
   ```bash
   cat logs/main_*.log
   ```

4. **提交结果到比赛平台**
   - 文件路径：`prediction_result/final_result.csv`
   - 比赛链接：https://tianchi.aliyun.com/competition/entrance/531842

5. **撰写实验报告**
   - 参考日志中的统计数据
   - 包含数据分析、模型设计、训练过程、结果展示
   - 解释loss函数原理

---

## 📧 联系方式

- 课程邮箱: data_science_2025@163.com
- 比赛论坛: https://tianchi.aliyun.com/competition/entrance/531842/forum

---

**祝实验顺利！🎉**
