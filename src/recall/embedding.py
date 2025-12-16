"""
基于Embedding的召回算法

使用文章的embedding向量，通过faiss进行快速相似度检索
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging
import pickle
import os

logger = logging.getLogger()


def embedding_sim(item_emb_df, save_path='user_data/emb_i2i_sim.pkl', topk=20):
    """
    使用faiss计算文章embedding相似度

    Args:
        item_emb_df: 文章embedding DataFrame
        save_path: 保存路径
        topk: 每个文章保留最相似的topk个文章

    Returns:
        emb_i2i_sim: {item_id: {similar_item_id: similarity}}
    """
    try:
        import faiss
    except:
        logger.warning('faiss未安装，使用numpy计算相似度（较慢）')
        return embedding_sim_numpy(item_emb_df, save_path, topk)

    logger.info('开始计算Embedding相似度矩阵（使用faiss）...')

    # 准备embedding矩阵
    item_ids = item_emb_df['article_id'].values
    emb_cols = [col for col in item_emb_df.columns if col.startswith('emb_')]
    emb_matrix = item_emb_df[emb_cols].values.astype('float32')
    
    # 确保数组是C连续的（faiss要求）
    emb_matrix = np.ascontiguousarray(emb_matrix)

    # 归一化
    faiss.normalize_L2(emb_matrix)

    # 构建索引
    logger.info('构建faiss索引...')
    index = faiss.IndexFlatIP(emb_matrix.shape[1])  # 内积相似度
    index.add(emb_matrix)

    # 搜索最相似的topk+1个（包括自己）
    logger.info('搜索相似文章...')
    similarities, indices = index.search(emb_matrix, topk + 1)

    # 构建相似度字典
    emb_i2i_sim = {}
    for i, item_id in enumerate(item_ids):
        emb_i2i_sim[item_id] = {}
        for j in range(1, len(indices[i])):  # 跳过自己（index 0）
            sim_item_id = item_ids[indices[i][j]]
            sim_score = float(similarities[i][j])
            emb_i2i_sim[item_id][sim_item_id] = sim_score

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(emb_i2i_sim, f)

    logger.info(f'Embedding相似度矩阵计算完成，已保存至: {save_path}')
    logger.info(f'文章总数: {len(emb_i2i_sim)}')

    return emb_i2i_sim


def embedding_sim_numpy(item_emb_df, save_path, topk=20):
    """
    使用numpy计算embedding相似度（备用方案）
    """
    logger.info('使用numpy计算Embedding相似度（可能较慢）...')

    item_ids = item_emb_df['article_id'].values
    emb_cols = [col for col in item_emb_df.columns if col.startswith('emb_')]
    emb_matrix = item_emb_df[emb_cols].values

    # 归一化
    emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

    # 计算相似度（分块计算避免内存溢出）
    emb_i2i_sim = {}
    batch_size = 1000

    for i in tqdm(range(0, len(item_ids), batch_size), desc='计算相似度'):
        batch_end = min(i + batch_size, len(item_ids))
        batch_sim = np.dot(emb_matrix[i:batch_end], emb_matrix.T)

        for j in range(batch_end - i):
            item_id = item_ids[i + j]
            # 取topk相似
            sim_scores = batch_sim[j]
            topk_indices = np.argsort(sim_scores)[::-1][1:topk+1]  # 跳过自己

            emb_i2i_sim[item_id] = {}
            for idx in topk_indices:
                emb_i2i_sim[item_id][item_ids[idx]] = float(sim_scores[idx])

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(emb_i2i_sim, f)

    logger.info(f'Embedding相似度矩阵计算完成')
    return emb_i2i_sim


def embedding_recall_batch(click_df, emb_i2i_sim, top_k=10, item_num=50):
    """
    基于Embedding的批量召回

    Args:
        click_df: 点击日志
        emb_i2i_sim: embedding相似度矩阵
        top_k: 每个历史物品召回的相似物品数
        item_num: 最终每个用户召回的物品数

    Returns:
        user_recall_items_dict: {user_id: [(item_id, score), ...]}
    """
    logger.info('开始Embedding批量召回...')

    # 构建用户点击历史
    from .itemcf import get_user_item_time
    user_item_time_dict = get_user_item_time(click_df)

    # 对每个用户进行召回
    user_recall_items_dict = {}
    for user_id in tqdm(user_item_time_dict.keys(), desc='Embedding召回'):
        user_hist_items = user_item_time_dict[user_id]
        user_hist_items_ = [item_id for item_id, _ in user_hist_items]

        item_scores = defaultdict(float)

        # 对于用户历史中的每个物品
        for loc, (item_i, click_time) in enumerate(user_hist_items):
            if item_i not in emb_i2i_sim:
                continue

            # 位置权重
            loc_weight = 0.9 ** (len(user_hist_items) - loc - 1)

            # 获取相似物品
            sim_items = sorted(emb_i2i_sim[item_i].items(),
                             key=lambda x: x[1], reverse=True)[:top_k]

            for item_j, sim_score in sim_items:
                if item_j in user_hist_items_:
                    continue
                item_scores[item_j] += sim_score * loc_weight

        # 排序
        recall_items = sorted(item_scores.items(),
                            key=lambda x: x[1], reverse=True)[:item_num]
        user_recall_items_dict[user_id] = recall_items

    logger.info(f'Embedding召回完成，用户数: {len(user_recall_items_dict)}')
    return user_recall_items_dict
