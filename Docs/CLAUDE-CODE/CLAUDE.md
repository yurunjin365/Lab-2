# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a news recommendation system competition project (天池 - 零基础入门推荐系统 - 新闻推荐). The task is to predict which news articles users will click next based on their historical click behavior.

**Competition Background:**
- Task: Predict the last clicked article for each user based on historical click logs
- Dataset: ~300K users, ~3M clicks, ~360K unique articles
- Training set: 200K users
- Test set A: 50K users (completely different from training users)
- Evaluation metric: MRR (Mean Reciprocal Rank)

**Key Insight:** Training and test users are completely disjoint (train: user_id 0-199999, test: user_id 200000-249999), so test data must be included when computing statistics and similarities for online submission.

## Directory Structure

```
Lab-2/
├── Data/                    # Raw competition data (not tracked in git)
│   ├── train_click_log.csv     # Training user click logs
│   ├── testA_click_log.csv     # Test set A click logs
│   ├── articles.csv            # Article metadata
│   └── articles_emb.csv        # Pre-computed article embeddings (250-dim)
├── Docs/                    # Competition documentation and baseline guides
│   ├── introduce.md            # Competition rules and submission format
│   ├── core-requirement.md     # Lab requirements and grading criteria
│   ├── DataA1121.md           # Data download links
│   └── Baseline-Task1~5/       # Step-by-step baseline tutorials
├── src/                     # Source code (currently empty, to be developed)
├── results/                 # Output directory for predictions
│   └── sample_submit.csv       # Sample submission format
└── install_claude.sh        # Installation script
```

## Data Format

### train_click_log.csv / testA_click_log.csv
User click behavior logs:
- `user_id`: User unique identifier
- `click_article_id`: Article unique identifier
- `click_timestamp`: Click timestamp (ms)
- `click_environment`: Click environment (1/2/4, mostly 4)
- `click_deviceGroup`: Device group (1-5, mostly 1 and 3)
- `click_os`: Operating system (2/12/17/20)
- `click_country`: Country code
- `click_region`: Region code
- `click_referrer_type`: Referrer type (1-7)

### articles.csv
Article metadata:
- `article_id`: Article unique identifier
- `category_id`: Article category (461 unique categories)
- `created_at_ts`: Article creation timestamp (ms)
- `words_count`: Article word count

### articles_emb.csv
Pre-computed 250-dimensional embeddings for articles (295,141 articles have embeddings).

## Submission Format

File: `results/result.csv`

Format:
```csv
user_id,article_1,article_2,article_3,article_4,article_5
200000,123,456,789,234,567
200001,890,345,678,901,234
```

Each row contains a user_id and 5 recommended article_ids ranked by predicted click probability (highest to lowest).

## Common Development Workflow

### 1. Data Loading Modes

There are three development modes:

**Debug Mode** - Quick prototyping with sampled data:
```python
# Sample 10K users from training set
all_click = pd.read_csv('Data/train_click_log.csv')
sample_user_ids = np.random.choice(all_click.user_id.unique(), size=10000, replace=False)
all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
```

**Offline Validation Mode** - Model selection and parameter tuning:
```python
# Use only training data, split into train/val
all_click = pd.read_csv('Data/train_click_log.csv')
```

**Online Submission Mode** - Final predictions for test set:
```python
# Combine train + test data (required because test users are new)
trn_click = pd.read_csv('Data/train_click_log.csv')
tst_click = pd.read_csv('Data/testA_click_log.csv')
all_click = pd.concat([trn_click, tst_click])
```

### 2. Memory Optimization

Always use memory reduction for large DataFrames:
```python
def reduce_mem(df):
    """Reduce DataFrame memory usage by downcasting numeric types"""
    # Implementation in Docs/Baseline-Task1~5/Task1.md
```

### 3. Typical Pipeline

The standard recommendation system pipeline has two stages:

**Stage 1: Multi-channel Recall** - Quickly retrieve ~50-100 candidates per user
- ItemCF (Item-based Collaborative Filtering)
- UserCF (User-based Collaborative Filtering)
- Hot news recall
- Embedding-based similarity recall
- Time-based decay weighting

**Stage 2: Ranking** - Rank candidates and select top 5
- Feature engineering (user features, article features, cross features)
- Model training (LightGBM, XGBoost, DeepFM, etc.)
- Model ensembling

## Important Data Characteristics

From the data analysis (Docs/DataA1121.md), these insights are critical for feature engineering:

1. **User-article cold start**: Test users never appear in training data
2. **Minimum clicks**: Train users click ≥2 articles; test users may click only 1
3. **Repeat clicks exist**: Some users click the same article multiple times
4. **Stable environments**: Most users have consistent click_environment/device/os
5. **User activity variance**: Click counts range from 2 to 241 (use for activity features)
6. **Article popularity variance**: Some articles clicked 2500+ times, others only once
7. **Strong article correlation**: Users tend to click similar/related articles sequentially
8. **Category preferences**: Users show clear category preferences (can click 1-95 different categories)
9. **Word count preferences**: Users show preferences for article length (avg 200 words)
10. **Temporal patterns**: Time gaps between clicks vary significantly by user

## Key Implementation Notes

### ItemCF Similarity Calculation
The baseline uses item-based collaborative filtering with time decay:
```python
# Co-occurrence weighted by user history length
i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

# Normalize by item popularity
i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
```

### Evaluation Metric (MRR)
Mean Reciprocal Rank calculation:
```
MRR = (1/|Users|) * Σ(1/rank_i)
```
where rank_i is the position (1-5) of the true clicked article in recommendations. If not in top 5, score = 0.

### User-Item-Time Dictionary Format
Critical data structure used throughout:
```python
{
    user_id: [(article_id, timestamp), (article_id, timestamp), ...],
    ...
}
```
Sorted by timestamp ascending.

## Lab Requirements

Per `Docs/core-requirement.md`, the lab report must include:

1. **Data Analysis**: Complete data processing pipeline from raw data to model input
2. **Model Selection**: Justification and explanation of chosen models
3. **Training Process**: Hyperparameter tuning, comparing multiple models
4. **Loss Function**: Explain why minimizing this loss achieves the prediction goal
5. **Results**: Show final predictions with input/output examples
6. **Code + Logs**: All code must be runnable with timestamped logs showing intermediate results
7. **Team Collaboration**: Clear division of work among 1-2 members
8. **Personal Reflection**: Individual summary of learnings

**Critical**: Code and logs must be original to avoid plagiarism detection.

## Baseline References

Detailed baseline implementations are in `Docs/Baseline-Task1~5/`:
- Task1: Problem understanding + ItemCF baseline
- Task2: Data analysis and visualization
- Task3: Multi-channel recall strategies
- Task4: Feature engineering
- Task5: Ranking model training

## Development Best Practices

1. **Always work with full data for submission**: Test users are cold-start, so compute item-item similarities and statistics using train+test data
2. **Use timestamp sorting**: Click sequences must be sorted by `click_timestamp`
3. **Handle missing embeddings**: Only 295K/364K articles have embeddings
4. **Optimize memory**: Data files are large (articles_emb.csv is 973MB)
5. **Log everything**: Timestamp all intermediate results for lab report
6. **Version control**: Track experiments and hyperparameters systematically
