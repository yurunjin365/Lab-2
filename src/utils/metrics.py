"""
评估指标计算
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger()


def mrr_score(y_true, y_pred):
    """
    计算MRR (Mean Reciprocal Rank) 评分

    MRR = 1/|Q| * Σ(1/rank_i)
    - 如果真实点击的文章在推荐列表中排名第1，得分为1
    - 如果真实点击的文章在推荐列表中排名第2，得分为0.5
    - 如果真实点击的文章在推荐列表中排名第3，得分为0.33
    - 如果真实点击的文章不在推荐列表中，得分为0

    Args:
        y_true: dict, {user_id: clicked_article_id}
        y_pred: dict, {user_id: [article_1, article_2, article_3, article_4, article_5]}

    Returns:
        mrr: MRR评分
    """
    scores = []
    for user_id in y_true:
        true_item = y_true[user_id]
        pred_items = y_pred.get(user_id, [])

        if true_item in pred_items:
            rank = pred_items.index(true_item) + 1  # 排名从1开始
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)

    mrr = np.mean(scores)
    return mrr


def hit_rate(y_true, y_pred, k=5):
    """
    计算Hit Rate@K：真实点击的文章是否在Top K推荐中

    Args:
        y_true: dict, {user_id: clicked_article_id}
        y_pred: dict, {user_id: [article_1, article_2, ..., article_k]}
        k: Top K

    Returns:
        hit_rate: Hit Rate@K
    """
    hits = 0
    for user_id in y_true:
        true_item = y_true[user_id]
        pred_items = y_pred.get(user_id, [])[:k]

        if true_item in pred_items:
            hits += 1

    hit_rate = hits / len(y_true) if len(y_true) > 0 else 0
    return hit_rate


def evaluate_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
    评估召回结果

    Args:
        user_recall_items_dict: {user_id: [(item_id, score), ...]}
        trn_last_click_df: 验证集最后一次点击记录
        topk: Top K

    Returns:
        mrr, hit_rate_5, hit_rate_10
    """
    # 准备真实标签
    y_true = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))

    # 准备预测结果
    y_pred = {}
    for user_id, item_scores in user_recall_items_dict.items():
        # item_scores是[(item_id, score), ...]列表，按score降序排列
        y_pred[user_id] = [item_id for item_id, _ in item_scores[:topk]]

    # 计算指标
    mrr = mrr_score(y_true, y_pred)
    hr5 = hit_rate(y_true, y_pred, k=5)
    hr10 = hit_rate(y_true, y_pred, k=10)

    logger.info(f'召回评估 - MRR: {mrr:.4f}, Hit@5: {hr5:.4f}, Hit@10: {hr10:.4f}')

    return mrr, hr5, hr10


def submit(recall_df, topk=5, model_name='baseline'):
    """
    生成提交文件

    Args:
        recall_df: 召回结果DataFrame，需包含user_id, click_article_id, pred_score列
        topk: Top K
        model_name: 模型名称

    Returns:
        提交文件路径
    """
    # 按用户分组，取Top K
    recall_df = recall_df.sort_values('pred_score', ascending=False)
    recall_df['rank'] = recall_df.groupby('user_id')['pred_score'].rank(ascending=False, method='first')

    # 只保留Top K
    sub_df = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    # 重命名列
    sub_df.columns = ['user_id'] + [f'article_{i}' for i in range(1, topk + 1)]

    # 保证测试集所有用户都有推荐结果
    # 如果某些用户没有召回结果，需要用热门文章填充
    # 这里先检查是否所有测试集用户都有推荐
    logger.info(f'提交文件用户数: {len(sub_df)}')

    # 保存
    import os
    os.makedirs('prediction_result', exist_ok=True)
    save_path = f'prediction_result/{model_name}_result.csv'
    sub_df.to_csv(save_path, index=False)

    logger.info(f'提交文件已保存: {save_path}')
    logger.info(f'文件格式: {sub_df.shape}')
    logger.info(f'前5行:\n{sub_df.head()}')

    return save_path
