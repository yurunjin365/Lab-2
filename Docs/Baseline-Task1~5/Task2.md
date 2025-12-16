**`[模式：文档撰写]` (Mode: Documentation)**

好的，我将采用 **`[模式：文档撰写]`** 来将这段格式混乱的文本转换为规范的 Markdown 格式文档。

---

# 数据分析

数据分析的价值主要在于熟悉了解整个数据集的基本情况，包括每个文件里有哪些数据，具体的文件中的每个字段表示什么实际含义，以及数据集中特征之间的相关性。在推荐场景下主要就是分析用户本身的基本属性，文章基本属性，以及用户和文章交互的一些分布，这些都有利于后面的召回策略的选择，以及特征工程。

**建议：** 当特征工程和模型调参已经很难继续上分了，可以回来重新从新的角度去分析这些数据，或许可以找到上分的灵感。

---

## 导包

```python
%matplotlib inline
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13)

import os, gc, re, warnings, sys
warnings.filterwarnings("ignore")
```

---

## 读取数据

```python
# path = './data/' # 自定义的路径
path = './' # 天池平台路径

##### train
trn_click = pd.read_csv(path + 'train_click_log.csv')
# trn_click = pd.read_csv(path + 'train_click_log.csv', names=['user_id','item_id','click_time','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type'])
item_df = pd.read_csv(path + 'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  # 重命名,方便后续match
item_emb_df = pd.read_csv(path + 'articles_emb.csv')

##### test
tst_click = pd.read_csv(path + 'testA_click_log.csv')
```

---

## 数据预处理

### 计算用户点击rank和点击次数

```python
# 对每个用户的点击时间戳进行排序
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
```

```python
# 计算用户点击文章的次数，并添加新的一列count
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')
```

---

## 数据浏览

### 用户点击日志文件_训练集

```python
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
trn_click.head()
```

**输出示例：**

| user_id | click_article_id | click_timestamp | click_environment | click_deviceGroup | click_os | click_country | click_region | click_referrer_type | rank | click_cnts | category_id | created_at_ts | words_count |
|---------|------------------|-----------------|-------------------|-------------------|----------|---------------|--------------|---------------------|------|------------|-------------|---------------|-------------|
| 199999  | 160417           | 1507029570190   | 4                 | 1                 | 17       | 1             | 13           | 1                   | 11   | 11         | 281         | 1506942089000 | 173         |
| 199999  | 5408             | 1507029571478   | 4                 | 1                 | 17       | 1             | 13           | 1                   | 10   | 11         | 4           | 1506994257000 | 118         |
| 199999  | 50823            | 1507029601478   | 4                 | 1                 | 17       | 1             | 13           | 1                   | 9    | 11         | 99          | 1507013614000 | 213         |
| 199998  | 157770           | 1507029532200   | 4                 | 1                 | 17       | 1             | 25           | 5                   | 40   | 40         | 281         | 1506983935000 | 201         |
| 199998  | 96613            | 1507029671831   | 4                 | 1                 | 17       | 1             | 25           | 5                   | 39   | 40         | 209         | 1506938444000 | 185         |

### train_click_log.csv 文件数据中每个字段的含义

- **user_id**: 用户的唯一标识
- **click_article_id**: 用户点击的文章唯一标识
- **click_timestamp**: 用户点击文章时的时间戳
- **click_environment**: 用户点击文章的环境
- **click_deviceGroup**: 用户点击文章的设备组
- **click_os**: 用户点击文章时的操作系统
- **click_country**: 用户点击文章时所在的国家
- **click_region**: 用户点击文章时所在的区域
- **click_referrer_type**: 用户点击文章时，文章的来源

```python
# 用户点击日志信息
trn_click.info()
```

**输出：**
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1112623 entries, 0 to 1112622
Data columns (total 14 columns):
user_id                1112623 non-null int64
click_article_id       1112623 non-null int64
click_timestamp        1112623 non-null int64
click_environment      1112623 non-null int64
click_deviceGroup      1112623 non-null int64
click_os               1112623 non-null int64
click_country          1112623 non-null int64
click_region           1112623 non-null int64
click_referrer_type    1112623 non-null int64
rank                   1112623 non-null int64
click_cnts             1112623 non-null int64
category_id            1112623 non-null int64
created_at_ts          1112623 non-null int64
words_count            1112623 non-null int64
dtypes: int64(14)
memory usage: 127.3 MB
```

```python
trn_click.describe()
```

**统计描述输出：**

| 统计量 | user_id | click_article_id | click_timestamp | click_environment | click_deviceGroup | click_os | click_country | click_region | click_referrer_type | rank | click_cnts | category_id | created_at_ts | words_count |
|--------|---------|------------------|-----------------|-------------------|-------------------|----------|---------------|--------------|---------------------|------|------------|-------------|---------------|-------------|
| count  | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 | 1.112623e+06 |
| mean   | 1.221198e+05 | 1.951541e+05 | 1.507588e+12 | 3.947786e+00 | 1.815981e+00 | 1.301976e+01 | 1.310776e+00 | 1.813587e+01 | 1.910063e+00 | 7.118518e+00 | 1.323704e+01 | 3.056176e+02 | 1.506598e+12 | 2.011981e+02 |
| std    | 5.540349e+04 | 9.292286e+04 | 3.363466e+08 | 3.276715e-01 | 1.035170e+00 | 6.967844e+00 | 1.618264e+00 | 7.105832e+00 | 1.220012e+00 | 1.016095e+01 | 1.631503e+01 | 1.155791e+02 | 8.343066e+09 | 5.223881e+01 |
| min    | 0.000000e+00 | 3.000000e+00 | 1.507030e+12 | 1.000000e+00 | 1.000000e+00 | 2.000000e+00 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 | 2.000000e+00 | 1.000000e+00 | 1.166573e+12 | 0.000000e+00 |
| 25%    | 7.934700e+04 | 1.239090e+05 | 1.507297e+12 | 4.000000e+00 | 1.000000e+00 | 2.000000e+00 | 1.000000e+00 | 1.300000e+01 | 1.000000e+00 | 2.000000e+00 | 4.000000e+00 | 2.500000e+02 | 1.507220e+12 | 1.700000e+02 |
| 50%    | 1.309670e+05 | 2.038900e+05 | 1.507596e+12 | 4.000000e+00 | 1.000000e+00 | 1.700000e+01 | 1.000000e+00 | 2.100000e+01 | 2.000000e+00 | 4.000000e+00 | 8.000000e+00 | 3.280000e+02 | 1.507553e+12 | 1.970000e+02 |
| 75%    | 1.704010e+05 | 2.777120e+05 | 1.507841e+12 | 4.000000e+00 | 3.000000e+00 | 1.700000e+01 | 1.000000e+00 | 2.500000e+01 | 2.000000e+00 | 8.000000e+00 | 1.600000e+01 | 4.100000e+02 | 1.507756e+12 | 2.280000e+02 |
| max    | 1.999990e+05 | 3.640460e+05 | 1.510603e+12 | 4.000000e+00 | 5.000000e+00 | 2.000000e+01 | 1.100000e+01 | 2.800000e+01 | 7.000000e+00 | 2.410000e+02 | 2.410000e+02 | 4.600000e+02 | 1.510666e+12 | 6.690000e+03 |

```python
# 训练集中的用户数量为20w
trn_click.user_id.nunique()
```

**输出：** `200000`

```python
trn_click.groupby('user_id')['click_article_id'].count().min()  # 训练集里面每个用户至少点击了两篇文章
```

**输出：** `2`

---

### 画直方图大体看一下基本的属性分布

```python
plt.figure()
plt.figure(figsize=(15, 20))
i = 1
for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 
            'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
    plot_envs = plt.subplot(5, 2, i)
    i += 1
    v = trn_click[col].value_counts().reset_index()[:10]
    fig = sns.barplot(x=v['index'], y=v[col])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.title(col)
plt.tight_layout()
plt.show()
```

**注：** 此处 `click_cnts` 直方图表示的是每篇文章对应用户的点击次数累计图。也可以以用户角度分析，画出每个用户点击文章次数的直方图。

---

### 点击环境分析

```python
trn_click['click_environment'].value_counts()
```

**输出：**
```
4    1084627
2      25894
1       2102
Name: click_environment, dtype: int64
```

**分析：** 从点击环境 `click_environment` 来看：
- 仅有 2102 次（占 0.19%）点击环境为 1
- 仅有 25894 次（占 2.3%）点击环境为 2
- 剩余（占 97.6%）点击环境为 4

---

### 点击设备组分析

```python
trn_click['click_deviceGroup'].value_counts()
```

**输出：**
```
1    678187
3    395558
4     38731
5       141
2         6
Name: click_deviceGroup, dtype: int64
```

**分析：** 从点击设备组 `click_deviceGroup` 来看，设备1占大部分（61%），设备3占36%。

---

## 测试集用户点击日志

```python
tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
tst_click.head()
```

**输出示例：**

| user_id | click_article_id | click_timestamp | click_environment | click_deviceGroup | click_os | click_country | click_region | click_referrer_type | rank | click_cnts | category_id | created_at_ts | words_count |
|---------|------------------|-----------------|-------------------|-------------------|----------|---------------|--------------|---------------------|------|------------|-------------|---------------|-------------|
| 249999  | 160974           | 1506959142820   | 4                 | 1                 | 17       | 1             | 13           | 2                   | 19   | 19         | 281         | 1506912747000 | 259         |
| 249999  | 160417           | 1506959172820   | 4                 | 1                 | 17       | 1             | 13           | 2                   | 18   | 19         | 281         | 1506942089000 | 173         |
| 249998  | 160974           | 1506959056066   | 4                 | 1                 | 12       | 1             | 13           | 2                   | 5    | 5          | 281         | 1506912747000 | 259         |
| 249998  | 202557           | 1506959086066   | 4                 | 1                 | 12       | 1             | 13           | 2                   | 4    | 5          | 327         | 1506938401000 | 219         |
| 249997  | 183665           | 1506959088613   | 4                 | 1                 | 17       | 1             | 15           | 5                   | 7    | 7          | 301         | 1500895686000 | 256         |

```python
tst_click.describe()
```

**重要发现：**

我们可以看出训练集和测试集的用户是完全不一样的：
- 训练集的用户ID范围：0 ~ 199999
- 测试集A的用户ID范围：200000 ~ 249999

因此，在训练时，需要把测试集的数据也包括在内，称为**全量数据**。

**!!!!!!!!!!!!!!!后续将对训练集和测试集合并分析!!!!!!!!!!!**

```python
# 测试集中的用户数量为5w
tst_click.user_id.nunique()
```

**输出：** `50000`

```python
tst_click.groupby('user_id')['click_article_id'].count().min()  # 注意测试集里面有只点击过一次文章的用户
```

**输出：** `1`

---

## 新闻文章信息数据表

```python
# 新闻文章数据集浏览
item_df.head().append(item_df.tail())
```

**输出示例：**

| click_article_id | category_id | created_at_ts | words_count |
|------------------|-------------|---------------|-------------|
| 0                | 0           | 1513144419000 | 168         |
| 1                | 1           | 1405341936000 | 189         |
| 2                | 1           | 1408667706000 | 250         |
| 3                | 1           | 1408468313000 | 230         |
| 4                | 1           | 1407071171000 | 162         |
| 364042           | 460         | 1434034118000 | 144         |
| 364043           | 460         | 1434148472000 | 463         |
| 364044           | 460         | 1457974279000 | 177         |
| 364045           | 460         | 1515964737000 | 126         |
| 364046           | 460         | 1505811330000 | 479         |

```python
item_df['words_count'].value_counts()
```

```python
print(item_df['category_id'].nunique())  # 461个文章主题
item_df['category_id'].hist()
```

**输出：** `461`

```python
item_df.shape  # 364047篇文章
```

**输出：** `(364047, 4)`

---

## 新闻文章embedding向量表示

```python
item_emb_df.head()
```

**输出示例（5 rows × 251 columns）：**

| article_id | emb_0 | emb_1 | emb_2 | emb_3 | emb_4 | emb_5 | emb_6 | emb_7 | emb_8 | ... | emb_240 | emb_241 | emb_242 | emb_243 | emb_244 | emb_245 | emb_246 | emb_247 | emb_248 | emb_249 |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 0          | -0.161183 | -0.957233 | -0.137944 | 0.050855 | 0.830055 | 0.901365 | -0.335148 | -0.559561 | -0.500603 | ... | 0.321248 | 0.313999 | 0.636412 | 0.169179 | 0.540524 | -0.813182 | 0.286870 | -0.231686 | 0.597416 | 0.409623 |

```python
item_emb_df.shape
```

**输出：** `(295141, 251)`

---

## 数据分析

### 用户重复点击

```python
##### merge
user_click_merge = trn_click.append(tst_click)
```

```python
# 用户重复点击
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
user_click_count[:10]
```

**输出示例：**

| user_id | click_article_id | count |
|---------|------------------|-------|
| 0       | 30760            | 1     |
| 0       | 157507           | 1     |
| 1       | 63746            | 1     |
| 1       | 289197           | 1     |
| 2       | 36162            | 1     |
| 2       | 168401           | 1     |
| 3       | 36162            | 1     |
| 3       | 50644            | 1     |
| 4       | 39894            | 1     |
| 4       | 42567            | 1     |

```python
user_click_count[user_click_count['count'] > 7]
```

**输出：** 少量用户重复点击超过7次的记录

```python
user_click_count['count'].unique()
```

**输出：** `array([ 1,  2,  4,  3,  6,  5, 10,  7, 13])`

```python
# 用户点击新闻次数
user_click_count.loc[:,'count'].value_counts()
```

**输出：**
```
1     1605541
2       11621
3         422
4          77
5          26
6          12
10          4
7           3
13          1
Name: count, dtype: int64
```

**分析结论：** 有 1605541（约占 99.2%）的用户未重复阅读过文章，仅有极少数用户重复点击过某篇文章。这个也可以单独制作成特征。

---

### 用户点击环境变化分析

```python
def plot_envs(df, cols, r, c):
    plt.figure()
    plt.figure(figsize=(10, 5))
    i = 1
    for col in cols:
        plt.subplot(r, c, i)
        i += 1
        v = df[col].value_counts().reset_index()
        fig = sns.barplot(x=v['index'], y=v[col])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()
```

```python
# 分析用户点击环境变化是否明显，这里随机采样10个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=10, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
for _, user_df in sample_users.groupby('user_id'):
    plot_envs(user_df, cols, 2, 3)
```

**分析结论：** 可以看出绝大多数用户的点击环境是比较固定的。

**思路：** 可以基于这些环境的统计特征来代表该用户本身的属性。

---

### 用户点击新闻数量的分布

```python
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
plt.plot(user_click_item_count)
```

**分析：** 可以根据用户的点击文章次数看出用户的活跃度。

```python
# 点击次数在前50的用户
plt.plot(user_click_item_count[:50])
```

**结论：** 点击次数排前50的用户的点击次数都在100次以上。

**思路：** 我们可以定义点击次数大于等于100次的用户为活跃用户，这是一种简单的处理思路。判断用户活跃度，更加全面的是再结合上点击时间，后面我们会基于点击次数和点击时间两个方面来判断用户活跃度。

```python
# 点击次数排名在[25000:50000]之间
plt.plot(user_click_item_count[25000:50000])
```

**结论：** 可以看出点击次数小于等于两次的用户非常多，这些用户可以认为是非活跃用户。

---

### 新闻点击次数分析

```python
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
```

```python
plt.plot(item_click_count)
```

```python
plt.plot(item_click_count[:100])
```

**结论：** 可以看出点击次数最多的前100篇新闻，点击次数大于1000次。

```python
plt.plot(item_click_count[:20])
```

**结论：** 点击次数最多的前20篇新闻，点击次数大于2500。

**思路：** 可以定义这些新闻为热门新闻，这个也是简单的处理方式，后面我们也是根据点击次数和时间进行文章热度的一个划分。

```python
plt.plot(item_click_count[3500:])
```

**结论：** 可以发现很多新闻只被点击过一两次。

**思路：** 可以定义这些新闻是冷门新闻。

---

### 新闻共现频次：两篇新闻连续出现的次数

```python
tmp = user_click_merge.sort_values('click_timestamp')
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x: x.shift(-1))
union_item = tmp.groupby(['click_article_id', 'next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
union_item[['count']].describe()
```

**统计结果：**

| 统计量 | count |
|--------|-------|
| count  | 433597.000000 |
| mean   | 3.184139 |
| std    | 18.851753 |
| min    | 1.000000 |
| 25%    | 1.000000 |
| 50%    | 1.000000 |
| 75%    | 2.000000 |
| max    | 2202.000000 |

**分析：** 由统计数据可以看出，平均共现次数3.18，最高为2202，说明用户看的新闻相关性是比较强的。

```python
# 画个图直观地看一看
x = union_item['click_article_id']
y = union_item['count']
plt.scatter(x, y)
```

```python
plt.plot(union_item['count'].values[40000:])
```

**结论：** 大概有75000个pair至少共现一次。

---

### 新闻文章信息

```python
# 不同类型的新闻出现的次数
plt.plot(user_click_merge['category_id'].value_counts().values)
```

```python
# 出现次数比较少的新闻类型，有些新闻类型，基本上就出现过几次
plt.plot(user_click_merge['category_id'].value_counts().values[150:])
```

```python
# 新闻字数的描述性统计
user_click_merge['words_count'].describe()
```

**输出：**
```
count    1.630633e+06
mean     2.043012e+02
std      6.382198e+01
min      0.000000e+00
25%      1.720000e+02
50%      1.970000e+02
75%      2.290000e+02
max      6.690000e+03
Name: words_count, dtype: float64
```

```python
plt.plot(user_click_merge['words_count'].values)
```

---

### 用户点击的新闻类型的偏好

此特征可以用于度量用户的兴趣是否广泛。

```python
plt.plot(sorted(user_click_merge.groupby('user_id')['category_id'].nunique(), reverse=True))
```

**分析：** 从上图中可以看出有一小部分用户阅读类型是极其广泛的，大部分人都处在20个新闻类型以下。

```python
user_click_merge.groupby('user_id')['category_id'].nunique().reset_index().describe()
```

**统计结果：**

| 统计量 | user_id | category_id |
|--------|---------|-------------|
| count  | 250000.000000 | 250000.000000 |
| mean   | 124999.500000 | 4.573188 |
| std    | 72168.927986 | 4.419800 |
| min    | 0.000000 | 1.000000 |
| 25%    | 62499.750000 | 2.000000 |
| 50%    | 124999 | 3.000000 |
| 75%	 | 187499.250000 |	6.000000 |
| max	 | 249999.000000 |	95.000000 |

---

### 用户查看文章的长度的分布

通过统计不同用户点击新闻的平均字数，这个可以反映用户是对长文更感兴趣还是对短文更感兴趣。

```python
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True))
```

**分析：** 从上图中可以发现有一小部分人看的文章平均词数非常高，也有一小部分人看的平均文章次数非常低。大多数人偏好于阅读字数在200-400字之间的新闻。

```python
# 挑出大多数人的区间仔细看看
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)[1000:45000])
```

**分析：** 可以发现大多数人都是看250字以下的文章。

```python
# 更加详细的参数
user_click_merge.groupby('user_id')['words_count'].mean().reset_index().describe()
```

**统计结果：**

| 统计量 | user_id | words_count |
|--------|---------|-------------|
| count  | 250000.000000 | 250000.000000 |
| mean   | 124999.500000 | 205.830189 |
| std    | 72168.927986 | 47.174030 |
| min    | 0.000000 | 8.000000 |
| 25%    | 62499.750000 | 187.500000 |
| 50%    | 124999.500000 | 202.000000 |
| 75%    | 187499.250000 | 217.750000 |
| max    | 249999.000000 | 3434.500000 |

---

### 用户点击新闻的时间分析

```python
# 为了更好的可视化，这里把时间进行归一化操作
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])

user_click_merge = user_click_merge.sort_values('click_timestamp')
```

```python
user_click_merge.head()
```

**输出示例：**

| user_id | click_article_id | click_timestamp | click_environment | click_deviceGroup | click_os | click_country | click_region | click_referrer_type | rank | click_cnts | category_id | created_at_ts | words_count |
|---------|------------------|-----------------|-------------------|-------------------|----------|---------------|--------------|---------------------|------|------------|-------------|---------------|-------------|
| 249990  | 162300           | 0.000000        | 4                 | 3                 | 20       | 1             | 25           | 2                   | 5    | 5          | 281         | 0.989186      | 193         |
| 249998  | 160974           | 0.000002        | 4                 | 1                 | 12       | 1             | 13           | 2                   | 5    | 5          | 281         | 0.989092      | 259         |
| 249985  | 160974           | 0.000003        | 4                 | 1                 | 17       | 1             | 8            | 2                   | 8    | 8          | 281         | 0.989092      | 259         |
| 249979  | 162300           | 0.000004        | 4                 | 1                 | 17       | 1             | 25           | 2                   | 2    | 2          | 281         | 0.989186      | 193         |
| 249988  | 160974           | 0.000004        | 4                 | 1                 | 17       | 1             | 21           | 2                   | 17   | 17         | 281         | 0.989092      | 259         |

```python
def mean_diff_time_func(df, col):
    df = pd.DataFrame(df, columns={col})
    df['time_shift1'] = df[col].shift(1).fillna(0)
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()
```

```python
# 点击时间差的平均值
mean_diff_click_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))
```

```python
plt.plot(sorted(mean_diff_click_time.values, reverse=True))
```

**分析：** 从上图可以发现不同用户点击文章的时间差是有差异的。

```python
# 前后点击文章的创建时间差的平均值
mean_diff_created_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'created_at_ts'))
```

```python
plt.plot(sorted(mean_diff_created_time.values, reverse=True))
```

**分析：** 从图中可以发现用户先后点击文章，文章的创建时间也是有差异的。

---

### Word2Vec 文章向量训练与可视化

#### 安装依赖

```bash
!pip install gensim
```

**输出：**
```
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://mirrors.aliyun.com/pypi/simple
Collecting gensim
  Downloading https://mirrors.aliyun.com/pypi/packages/2b/e0/fa6326251692056dc880a64eb22117e03269906ba55a6864864d24ec8b4e/gensim-3.8.3-cp36-cp36m-manylinux1_x86_64.whl (24.2 MB)
     |████████████████████████████████| 24.2 MB 91.0 MB/s eta 0:00:01
Requirement already satisfied: six>=1.5.0 in /opt/conda/lib/python3.6/site-packages (from gensim) (1.15.0)
Requirement already satisfied: numpy>=1.11.3 in /opt/conda/lib/python3.6/site-packages (from gensim) (1.19.1)
Requirement already satisfied: scipy>=0.18.1 in /opt/conda/lib/python3.6/site-packages (from gensim) (1.5.4)
Collecting smart-open>=1.8.1
  Downloading https://mirrors.aliyun.com/pypi/packages/e3/cf/6311dfb0aff3e295d63930dea72e3029800242cdfe0790478e33eccee2ab/smart_open-4.0.1.tar.gz (117 kB)
     |████████████████████████████████| 117 kB 96.7 MB/s eta 0:00:01
Building wheels for collected packages: smart-open
  Building wheel for smart-open (setup.py) ... done
Successfully built smart-open
Installing collected packages: smart-open, gensim
Successfully installed gensim-3.8.3 smart-open-4.0.1
```

#### 训练 Word2Vec 模型

```python
from gensim.models import Word2Vec
import logging, pickle

# 需要注意这里模型只迭代了一次
def trian_item_word2vec(click_df, embed_size=16, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大，默认负采样为5
    w2v = Word2Vec(docs, size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=10)
  
    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v[k] for k in click_df['click_article_id']}
  
    return item_w2v_emb_dict
```

```python
item_w2v_emb_dict = trian_item_word2vec(user_click_merge)
```

#### 文章相似度可视化

```python
# 随机选择5个用户，查看这些用户前后查看文章的相似性
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

sub_user_info.head()
```

**输出示例：**

| user_id | click_article_id | click_timestamp | click_environment | click_deviceGroup | click_os | click_country | click_region | click_referrer_type |
|---------|------------------|-----------------|-------------------|-------------------|----------|---------------|--------------|---------------------|
| 190841  | 199197           | 1507045276129   | 4                 | 1                 | 17       | 1             | 20           | 2                   |
| 190841  | 285298           | 1507045302920   | 4                 | 1                 | 17       | 1             | 20           | 2                   |
| 190841  | 156624           | 1507046638885   | 4                 | 1                 | 17       | 1             | 20           | 2                   |
| 190841  | 129029           | 1507046668885   | 4                 | 1                 | 17       | 1             | 20           | 2                   |
| 164226  | 214800           | 1507131402464   | 4                 | 1                 | 17       | 1             | 21           | 2                   |

```python
# 上一个版本，这个函数使用的是赛题提供的词向量，但是由于给出的embedding并不是所有的数据的embedding，所以运行下面画图函数的时候会报keyerror的错误
# 为了防止出现这个错误，这里修改为使用word2vec训练得到的词向量进行可视化
def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_w2v_emb_dict[str(item_list[i])]  # 需要注意的是word2vec训练时候使用的是str类型的数据
        emb2 = item_w2v_emb_dict[str(item_list[i+1])]
        sim_list.append(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2))))
    sim_list.append(0)
    return sim_list
```

```python
for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)
```

**注意：** 这里由于对词向量的训练迭代次数不是很多，所以看到的可视化结果不是很准确，可以训练更多次来观察具体的现象。

---

## 总结

通过数据分析的过程，我们目前可以得到以下几点重要的信息，这个对于我们进行后面的特征制作和分析非常有帮助：

1. **训练集和测试集的用户ID没有重复**，也就是测试集里面的用户模型是没有见过的
2. **训练集中用户最少的点击文章数是2**，而测试集里面用户最少的点击文章数是1
3. 用户对于文章**存在重复点击的情况**，但这个都存在于训练集里面
4. 同一用户的点击环境**存在不唯一的情况**，后面做这部分特征的时候可以采用统计特征
5. **用户点击文章的次数有很大的区分度**，后面可以根据这个制作衡量用户活跃度的特征
6. **文章被用户点击的次数也有很大的区分度**，后面可以根据这个制作衡量文章热度的特征
7. **用户看的新闻，相关性是比较强的**，所以往往我们判断用户是否对某篇文章感兴趣的时候，在很大程度上会和他历史点击过的文章有关
8. **用户点击的文章字数有比较大的区别**，这个可以反映用户对于文章字数的偏好
9. **用户点击过的文章主题也有很大的区别**，这个可以反映用户的主题偏好
10. **不同用户点击文章的时间差也会有所区别**，这个可以反映用户对于文章时效性的偏好

所以根据上面的一些分析，可以更好的帮助我们后面做好特征工程，充分挖掘数据的隐含信息。

---

**文档转换完成！** 

这份完整的 Markdown 文档已经将原始混乱的文本进行了规范化处理，包括：
- 清晰的标题层级结构
- 正确格式化的代码块（带语言标识）
- 规范的表格呈现
- 有序的章节组织
- 重点内容的标注和分析总结

如需进一步调整或补充，请告诉我。