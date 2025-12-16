"""
ItemCF (Item Collaborative Filtering) 召回算法

基于物品的协同过滤算法，通过计算物品之间的相似度来推荐用户可能感兴趣的物品。
核心思想：如果用户A和用户B都喜欢物品I，那么A喜欢的其他物品很可能也是B感兴趣的。

相似度计算考虑：
1. 时间衰减：用户最近点击的文章权重更高
2. 位置权重：用户点击序列中靠前的文章权重更高
3. 文章创建时间：新文章权重更高
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging
import pickle
import math
import os

logger = logging.getLogger()


def itemcf_sim(all_click_df, item_created_time_dict=None, save_path='user_data/itemcf_i2i_sim.pkl'):
    """
    计算ItemCF物品相似度矩阵

    Args:
        all_click_df: 所有点击数据（训练集+测试集）
        item_created_time_dict: 文章创建时间字典 {item_id: created_time}
        save_path: 相似度矩阵保存路径

    Returns:
        i2i_sim: 物品相似度字典 {item_i: {item_j: similarity}}
    """
    logger.info('开始计算ItemCF相似度矩阵...')

    # 按用户分组，获取每个用户的点击序列
    user_item_time_dict = defaultdict(list)

    for idx, row in tqdm(all_click_df.iterrows(), total=len(all_click_df), desc='构建用户点击序列'):
        user_id = row['user_id']
        item_id = row['click_article_id']
        click_time = row['click_timestamp']
        user_item_time_dict[user_id].append((item_id, click_time))

    # 按时间排序
    for user_id in user_item_time_dict:
        user_item_time_dict[user_id] = sorted(user_item_time_dict[user_id], key=lambda x: x[1])

    # 计算物品之间的共现次数和相似度
    i2i_count = defaultdict(lambda: defaultdict(int))  # 共现次数
    item_cnt = defaultdict(int)  # 每个物品被点击的次数

    logger.info('计算物品共现矩阵...')
    for user_id, item_time_list in tqdm(user_item_time_dict.items(), desc='计算共现'):
        # 遍历该用户的点击序列
        for loc1, (item_i, time_i) in enumerate(item_time_list):
            item_cnt[item_i] += 1

            # 计算与后续点击物品的相似度
            for loc2, (item_j, time_j) in enumerate(item_time_list[loc1+1:]):
                # 位置权重：点击位置越靠前权重越高
                loc_weight = 0.9 ** (loc2 + 1)

                # 时间衰减：点击时间越近权重越高
                time_diff = (time_j - time_i) / 3600000  # 转为小时
                time_weight = 1.0 / (1.0 + time_diff)

                # 文章创建时间权重
                if item_created_time_dict:
                    created_i = item_created_time_dict.get(item_i, time_i)
                    created_j = item_created_time_dict.get(item_j, time_j)
                    # 新文章权重更高
                    create_weight = 1.0 if created_j > created_i else 0.7
                else:
                    create_weight = 1.0

                # 综合权重
                total_weight = loc_weight * time_weight * create_weight

                i2i_count[item_i][item_j] += total_weight

    # 计算相似度：使用余弦相似度
    logger.info('计算物品相似度...')
    i2i_sim = {}

    for item_i, related_items in tqdm(i2i_count.items(), desc='计算相似度'):
        i2i_sim[item_i] = {}
        for item_j, count in related_items.items():
            # 余弦相似度
            i2i_sim[item_i][item_j] = count / math.sqrt(item_cnt[item_i] * item_cnt[item_j])

    # 保存相似度矩阵
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(i2i_sim, f)

    logger.info(f'ItemCF相似度矩阵计算完成，已保存至: {save_path}')
    logger.info(f'物品总数: {len(i2i_sim)}')
    logger.info(f'平均每个物品的相似物品数: {np.mean([len(v) for v in i2i_sim.values()]):.2f}')

    return i2i_sim


def itemcf_recall(user_item_time_dict, i2i_sim, user_id, top_k=10, item_num=50):
    """
    基于ItemCF的召回

    Args:
        user_item_time_dict: 用户点击历史 {user_id: [(item_id, time), ...]}
        i2i_sim: 物品相似度矩阵
        user_id: 目标用户ID
        top_k: 对于用户历史中的每个物品，召回top_k个相似物品
        item_num: 最终返回的召回物品数量

    Returns:
        recall_items: [(item_id, score), ...] 按score降序排列
    """
    # 获取用户历史点击
    if user_id not in user_item_time_dict:
        return []

    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = [item_id for item_id, _ in user_hist_items]

    # 候选物品得分
    item_scores = defaultdict(float)

    # 对于用户历史中的每个物品
    for loc, (item_i, click_time) in enumerate(user_hist_items):
        # 获取相似物品
        if item_i not in i2i_sim:
            continue

        # 按相似度排序
        sim_items = sorted(i2i_sim[item_i].items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 位置权重：最近点击的物品权重更高
        loc_weight = 0.9 ** (len(user_hist_items) - loc - 1)

        for item_j, sim_score in sim_items:
            # 不推荐用户已经点击过的
            if item_j in user_hist_items_:
                continue

            # 累加得分
            item_scores[item_j] += sim_score * loc_weight

    # 排序并返回top item_num
    recall_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:item_num]

    return recall_items


def get_item_info_dict(item_info_df):
    """
    获取文章信息字典

    Args:
        item_info_df: 文章信息DataFrame

    Returns:
        item_created_time_dict: {item_id: created_time}
    """
    item_created_time_dict = dict(zip(item_info_df['click_article_id'],
                                      item_info_df['created_at_ts']))
    return item_created_time_dict


def get_user_item_time(click_df):
    """
    获取用户点击历史字典

    Args:
        click_df: 点击日志DataFrame

    Returns:
        user_item_time_dict: {user_id: [(item_id, time), ...]}
    """
    click_df = click_df.sort_values('click_timestamp')

    user_item_time_dict = defaultdict(list)
    for idx, row in click_df.iterrows():
        user_id = row['user_id']
        item_id = row['click_article_id']
        click_time = row['click_timestamp']
        user_item_time_dict[user_id].append((item_id, click_time))

    return user_item_time_dict


def itemcf_recall_batch(click_df, i2i_sim, top_k=10, item_num=50):
    """
    批量召回

    Args:
        click_df: 点击日志
        i2i_sim: 物品相似度矩阵
        top_k: 每个历史物品召回的相似物品数
        item_num: 最终每个用户召回的物品数

    Returns:
        user_recall_items_dict: {user_id: [(item_id, score), ...]}
    """
    logger.info('开始ItemCF批量召回...')

    # 构建用户点击历史
    user_item_time_dict = get_user_item_time(click_df)

    # 对每个用户进行召回
    user_recall_items_dict = {}
    for user_id in tqdm(user_item_time_dict.keys(), desc='ItemCF召回'):
        recall_items = itemcf_recall(user_item_time_dict, i2i_sim, user_id, top_k, item_num)
        user_recall_items_dict[user_id] = recall_items

    logger.info(f'ItemCF召回完成，用户数: {len(user_recall_items_dict)}')
    logger.info(f'平均每个用户召回物品数: {np.mean([len(v) for v in user_recall_items_dict.values()]):.2f}')

    return user_recall_items_dict


def hot_recall(click_df, top_k=50):
    """
    热门文章召回（作为冷启动策略）

    Args:
        click_df: 点击日志
        top_k: 返回最热门的k篇文章

    Returns:
        hot_items: [(item_id, click_count), ...]
    """
    logger.info('计算热门文章...')

    item_click_count = click_df['click_article_id'].value_counts().reset_index()
    item_click_count.columns = ['click_article_id', 'click_count']

    hot_items = list(zip(item_click_count['click_article_id'][:top_k].values,
                         item_click_count['click_count'][:top_k].values))

    logger.info(f'热门文章Top {top_k}: {hot_items[:10]}')

    return hot_items
