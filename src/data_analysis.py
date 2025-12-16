"""
数据探索性分析（EDA）脚本

分析内容：
1. 用户点击行为分析
2. 文章属性分析
3. 用户-文章交互分析
4. 时间特征分析
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from utils.data_loader import load_data, get_all_click_df, setup_logger

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logger = setup_logger('eda')


def analyze_user_click_behavior(all_click_df):
    """分析用户点击行为"""
    logger.info('\n========== 用户点击行为分析 ==========')

    # 1. 用户点击次数分布
    user_click_count = all_click_df.groupby('user_id')['click_article_id'].count()
    logger.info(f'用户点击次数统计:')
    logger.info(f'  最小值: {user_click_count.min()}')
    logger.info(f'  最大值: {user_click_count.max()}')
    logger.info(f'  平均值: {user_click_count.mean():.2f}')
    logger.info(f'  中位数: {user_click_count.median():.2f}')

    # 2. 用户重复点击分析
    user_repeat_click = all_click_df.groupby(['user_id', 'click_article_id']).size()
    repeat_count = (user_repeat_click > 1).sum()
    total_count = len(user_repeat_click)
    logger.info(f'\n用户重复点击分析:')
    logger.info(f'  重复点击次数: {repeat_count}')
    logger.info(f'  总点击组合数: {total_count}')
    logger.info(f'  重复点击比例: {repeat_count/total_count*100:.2f}%')

    # 3. 用户环境分析
    logger.info(f'\n用户点击环境分布:')
    for col in ['click_environment', 'click_deviceGroup', 'click_os']:
        logger.info(f'  {col}:')
        value_counts = all_click_df[col].value_counts()
        for val, count in value_counts.head().items():
            logger.info(f'    {val}: {count} ({count/len(all_click_df)*100:.2f}%)')

    # 4. 用户阅读主题多样性
    user_category_count = all_click_df.groupby('user_id')['category_id'].nunique()
    logger.info(f'\n用户阅读主题多样性:')
    logger.info(f'  平均阅读类别数: {user_category_count.mean():.2f}')
    logger.info(f'  中位数: {user_category_count.median():.2f}')
    logger.info(f'  最多类别数: {user_category_count.max()}')


def analyze_article_features(all_click_df, item_info_df):
    """分析文章属性"""
    logger.info('\n========== 文章属性分析 ==========')

    # 1. 文章点击次数分布
    item_click_count = all_click_df.groupby('click_article_id')['user_id'].count()
    logger.info(f'文章点击次数统计:')
    logger.info(f'  最小值: {item_click_count.min()}')
    logger.info(f'  最大值: {item_click_count.max()}')
    logger.info(f'  平均值: {item_click_count.mean():.2f}')
    logger.info(f'  中位数: {item_click_count.median():.2f}')

    # 热门文章
    logger.info(f'\nTop 10热门文章:')
    top10_items = item_click_count.sort_values(ascending=False).head(10)
    for item_id, count in top10_items.items():
        logger.info(f'  文章{item_id}: 点击{count}次')

    # 2. 文章类别分布
    category_count = item_info_df['category_id'].nunique()
    logger.info(f'\n文章类别统计:')
    logger.info(f'  总类别数: {category_count}')

    category_article_count = item_info_df.groupby('category_id').size()
    logger.info(f'  平均每类别文章数: {category_article_count.mean():.2f}')

    # 3. 文章字数分布
    logger.info(f'\n文章字数统计:')
    logger.info(f'  最小值: {item_info_df["words_count"].min()}')
    logger.info(f'  最大值: {item_info_df["words_count"].max()}')
    logger.info(f'  平均值: {item_info_df["words_count"].mean():.2f}')
    logger.info(f'  中位数: {item_info_df["words_count"].median():.2f}')


def analyze_user_article_interaction(all_click_df):
    """分析用户-文章交互"""
    logger.info('\n========== 用户-文章交互分析 ==========')

    # 1. 文章共现分析
    all_click_df_sorted = all_click_df.sort_values(['user_id', 'click_timestamp'])
    all_click_df_sorted['next_article'] = all_click_df_sorted.groupby('user_id')['click_article_id'].shift(-1)

    # 去除最后一次点击（没有next_article）
    co_occur = all_click_df_sorted[all_click_df_sorted['next_article'].notna()]

    # 统计共现次数
    co_occur_count = co_occur.groupby(['click_article_id', 'next_article']).size()
    logger.info(f'文章共现分析:')
    logger.info(f'  共现对数: {len(co_occur_count)}')
    logger.info(f'  平均共现次数: {co_occur_count.mean():.2f}')
    logger.info(f'  最大共现次数: {co_occur_count.max()}')

    # Top 10共现对
    logger.info(f'\nTop 10文章共现对:')
    top10_co = co_occur_count.sort_values(ascending=False).head(10)
    for (item1, item2), count in top10_co.items():
        logger.info(f'  文章{item1} -> 文章{item2}: {count}次')


def analyze_time_features(all_click_df):
    """分析时间特征"""
    logger.info('\n========== 时间特征分析 ==========')

    # 1. 点击时间跨度
    time_span = (all_click_df['click_timestamp'].max() - all_click_df['click_timestamp'].min()) / 3600000
    logger.info(f'数据时间跨度: {time_span:.2f}小时 ({time_span/24:.2f}天)')

    # 2. 用户点击时间间隔
    all_click_df_sorted = all_click_df.sort_values(['user_id', 'click_timestamp'])
    all_click_df_sorted['time_diff'] = all_click_df_sorted.groupby('user_id')['click_timestamp'].diff()

    # 转为小时
    time_diff_hours = all_click_df_sorted['time_diff'].dropna() / 3600000

    logger.info(f'\n用户点击时间间隔（小时）:')
    logger.info(f'  平均间隔: {time_diff_hours.mean():.2f}小时')
    logger.info(f'  中位数: {time_diff_hours.median():.2f}小时')

    # 3. 文章创建时间与点击时间差
    all_click_df['article_age'] = (all_click_df['click_timestamp'] - all_click_df['created_at_ts']) / 3600000
    logger.info(f'\n文章创建时间与点击时间差（文章年龄，小时）:')
    logger.info(f'  平均值: {all_click_df["article_age"].mean():.2f}小时')
    logger.info(f'  中位数: {all_click_df["article_age"].median():.2f}小时')


def check_train_test_overlap(trn_click, tst_click):
    """检查训练集和测试集的重叠情况"""
    logger.info('\n========== 训练集与测试集重叠分析 ==========')

    trn_users = set(trn_click['user_id'].unique())
    tst_users = set(tst_click['user_id'].unique())
    overlap_users = trn_users & tst_users

    logger.info(f'训练集用户数: {len(trn_users)}')
    logger.info(f'测试集用户数: {len(tst_users)}')
    logger.info(f'重叠用户数: {len(overlap_users)}')

    if len(overlap_users) == 0:
        logger.warning('⚠️ 训练集和测试集用户完全不重叠！这是冷启动问题！')
    else:
        logger.info(f'重叠比例: {len(overlap_users)/len(tst_users)*100:.2f}%')

    # 文章重叠
    trn_items = set(trn_click['click_article_id'].unique())
    tst_items = set(tst_click['click_article_id'].unique())
    overlap_items = trn_items & tst_items

    logger.info(f'\n训练集文章数: {len(trn_items)}')
    logger.info(f'测试集文章数: {len(tst_items)}')
    logger.info(f'重叠文章数: {len(overlap_items)}')
    logger.info(f'重叠比例: {len(overlap_items)/len(tst_items)*100:.2f}%')


def main_eda():
    """数据分析主流程"""
    logger.info('='*50)
    logger.info('开始数据探索性分析（EDA）')
    logger.info('='*50)

    start_time = datetime.now()

    # 加载数据
    logger.info('\n加载数据...')
    data_path = './Data/'
    trn_click, tst_click, item_info_df, item_emb_df = load_data(data_path)
    all_click_df = get_all_click_df(trn_click, tst_click, item_info_df)

    # 1. 用户点击行为分析
    analyze_user_click_behavior(all_click_df)

    # 2. 文章属性分析
    analyze_article_features(all_click_df, item_info_df)

    # 3. 用户-文章交互分析
    analyze_user_article_interaction(all_click_df)

    # 4. 时间特征分析
    analyze_time_features(all_click_df)

    # 5. 训练集-测试集重叠分析
    check_train_test_overlap(trn_click, tst_click)

    # 完成
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info('\n' + '='*50)
    logger.info(f'数据分析完成！总耗时: {elapsed:.2f}秒')
    logger.info('='*50)


if __name__ == '__main__':
    main_eda()
