"""
主流程：新闻推荐系统完整Pipeline

流程：
1. 数据加载
2. 多路召回（ItemCF + Embedding）
3. 特征工程
4. 排序模型训练
5. 模型融合
6. 生成提交文件
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import gc

# 导入自定义模块
from utils.data_loader import load_data, get_all_click_df, save_pickle, load_pickle, setup_logger
from utils.metrics import mrr_score, hit_rate, submit

from recall.itemcf import (itemcf_sim, itemcf_recall_batch, hot_recall,
                           get_item_info_dict, get_user_item_time)
from recall.embedding import embedding_sim, embedding_recall_batch

from features.feature_engineering import (create_user_features, create_item_features,
                                          make_train_set, make_test_set,
                                          make_candidate_set, add_features)

from models.lgb_ranker import (train_lgb_ranker, train_lgb_classifier,
                               predict_lgb, negative_sampling)


# 设置日志
logger = setup_logger('main')


def merge_recall_results(recall_dict_list, weights=None, topk=150):
    """
    融合多路召回结果

    Args:
        recall_dict_list: 多个召回结果列表 [{user_id: [(item, score), ...]}, ...]
        weights: 各路召回的权重，默认均等
        topk: 最终保留的候选数量

    Returns:
        merged_dict: 融合后的召回结果
    """
    logger.info('开始融合多路召回结果...')

    if weights is None:
        weights = [1.0] * len(recall_dict_list)

    # 归一化权重
    weights = np.array(weights) / sum(weights)

    merged_dict = {}

    # 获取所有用户
    all_users = set()
    for recall_dict in recall_dict_list:
        all_users.update(recall_dict.keys())

    logger.info(f'总用户数: {len(all_users)}')

    for user_id in all_users:
        item_scores = {}

        # 融合各路召回
        for recall_dict, weight in zip(recall_dict_list, weights):
            if user_id not in recall_dict:
                continue

            for item_id, score in recall_dict[user_id]:
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                # 加权累加
                item_scores[item_id] += score * weight

        # 排序并取topk
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        merged_dict[user_id] = sorted_items

    logger.info(f'召回融合完成，平均每用户候选数: {np.mean([len(v) for v in merged_dict.values()]):.2f}')

    return merged_dict


def fill_missing_users(user_recall_dict, all_user_ids, hot_items):
    """
    用热门文章填充没有召回结果的用户

    Args:
        user_recall_dict: 召回结果
        all_user_ids: 所有用户ID集合
        hot_items: 热门文章列表

    Returns:
        user_recall_dict: 填充后的召回结果
    """
    logger.info('填充缺失用户...')

    missing_users = set(all_user_ids) - set(user_recall_dict.keys())
    logger.info(f'缺失用户数: {len(missing_users)}')

    for user_id in missing_users:
        user_recall_dict[user_id] = hot_items[:50]

    return user_recall_dict


def main():
    """主流程"""
    logger.info('='*50)
    logger.info('开始运行新闻推荐系统')
    logger.info('='*50)

    start_time = datetime.now()

    # ==================== 1. 数据加载 ====================
    logger.info('\n【步骤1】数据加载')
    data_path = './Data/'

    trn_click, tst_click, item_info_df, item_emb_df = load_data(data_path)
    all_click_df = get_all_click_df(trn_click, tst_click, item_info_df)

    # 获取文章创建时间
    item_created_time_dict = get_item_info_dict(item_info_df)

    # 获取热门文章（用于冷启动）
    hot_items = hot_recall(all_click_df, top_k=50)

    # ==================== 2. 多路召回 ====================
    logger.info('\n【步骤2】多路召回')
    
    # 定义缓存路径
    itemcf_sim_path = 'user_data/itemcf_i2i_sim.pkl'
    emb_sim_path = 'user_data/emb_i2i_sim.pkl'
    trn_itemcf_recall_path = 'user_data/recall_results/trn_itemcf_recall.pkl'
    tst_itemcf_recall_path = 'user_data/recall_results/tst_itemcf_recall.pkl'
    trn_emb_recall_path = 'user_data/recall_results/trn_emb_recall.pkl'
    tst_emb_recall_path = 'user_data/recall_results/tst_emb_recall.pkl'
    trn_merged_recall_path = 'user_data/recall_results/trn_recall.pkl'
    tst_merged_recall_path = 'user_data/recall_results/tst_recall.pkl'

    # 2.1 ItemCF召回
    logger.info('\n--- ItemCF召回 ---')
    
    # 加载或计算ItemCF相似度矩阵
    if os.path.exists(itemcf_sim_path):
        logger.info('加载已有的ItemCF相似度矩阵...')
        with open(itemcf_sim_path, 'rb') as f:
            i2i_sim = pickle.load(f)
    else:
        logger.info('计算ItemCF相似度矩阵...')
        i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, itemcf_sim_path)

    # 加载或计算训练集ItemCF召回
    if os.path.exists(trn_itemcf_recall_path):
        logger.info('加载已有的训练集ItemCF召回结果...')
        trn_user_recall_itemcf = load_pickle(trn_itemcf_recall_path)
    else:
        logger.info('训练集召回（用于构建训练数据）...')
        trn_user_recall_itemcf = itemcf_recall_batch(trn_click, i2i_sim, top_k=10, item_num=100)
        save_pickle(trn_user_recall_itemcf, trn_itemcf_recall_path)
        logger.info(f'训练集ItemCF召回结果已缓存: {trn_itemcf_recall_path}')

    # 加载或计算测试集ItemCF召回
    if os.path.exists(tst_itemcf_recall_path):
        logger.info('加载已有的测试集ItemCF召回结果...')
        tst_user_recall_itemcf = load_pickle(tst_itemcf_recall_path)
    else:
        logger.info('测试集召回...')
        tst_user_recall_itemcf = itemcf_recall_batch(tst_click, i2i_sim, top_k=10, item_num=100)
        save_pickle(tst_user_recall_itemcf, tst_itemcf_recall_path)
        logger.info(f'测试集ItemCF召回结果已缓存: {tst_itemcf_recall_path}')

    # 2.2 Embedding召回
    logger.info('\n--- Embedding召回 ---')
    
    # 加载或计算Embedding相似度矩阵
    if os.path.exists(emb_sim_path):
        logger.info('加载已有的Embedding相似度矩阵...')
        with open(emb_sim_path, 'rb') as f:
            emb_i2i_sim = pickle.load(f)
    else:
        logger.info('计算Embedding相似度矩阵...')
        emb_i2i_sim = embedding_sim(item_emb_df, emb_sim_path, topk=20)

    # 加载或计算训练集Embedding召回
    if os.path.exists(trn_emb_recall_path):
        logger.info('加载已有的训练集Embedding召回结果...')
        trn_user_recall_emb = load_pickle(trn_emb_recall_path)
    else:
        logger.info('训练集Embedding召回...')
        trn_user_recall_emb = embedding_recall_batch(trn_click, emb_i2i_sim, top_k=10, item_num=100)
        save_pickle(trn_user_recall_emb, trn_emb_recall_path)
        logger.info(f'训练集Embedding召回结果已缓存: {trn_emb_recall_path}')

    # 加载或计算测试集Embedding召回
    if os.path.exists(tst_emb_recall_path):
        logger.info('加载已有的测试集Embedding召回结果...')
        tst_user_recall_emb = load_pickle(tst_emb_recall_path)
    else:
        logger.info('测试集Embedding召回...')
        tst_user_recall_emb = embedding_recall_batch(tst_click, emb_i2i_sim, top_k=10, item_num=100)
        save_pickle(tst_user_recall_emb, tst_emb_recall_path)
        logger.info(f'测试集Embedding召回结果已缓存: {tst_emb_recall_path}')

    # 2.3 融合召回
    logger.info('\n--- 融合召回 ---')
    
    # 检查是否有融合后的缓存
    if os.path.exists(trn_merged_recall_path) and os.path.exists(tst_merged_recall_path):
        logger.info('加载已有的融合召回结果...')
        trn_user_recall_merged = load_pickle(trn_merged_recall_path)
        tst_user_recall_merged = load_pickle(tst_merged_recall_path)
    else:
        trn_user_recall_merged = merge_recall_results(
            [trn_user_recall_itemcf, trn_user_recall_emb],
            weights=[0.6, 0.4],
            topk=150
        )

        tst_user_recall_merged = merge_recall_results(
            [tst_user_recall_itemcf, tst_user_recall_emb],
            weights=[0.6, 0.4],
            topk=150
        )

        # 填充缺失用户
        trn_user_recall_merged = fill_missing_users(trn_user_recall_merged,
                                                    trn_click['user_id'].unique(),
                                                    hot_items)
        tst_user_recall_merged = fill_missing_users(tst_user_recall_merged,
                                                    tst_click['user_id'].unique(),
                                                    hot_items)

        # 保存融合召回结果
        save_pickle(trn_user_recall_merged, trn_merged_recall_path)
        save_pickle(tst_user_recall_merged, tst_merged_recall_path)
        logger.info('融合召回结果已缓存')

    # ==================== 3. 离线验证划分 ====================
    logger.info('\n【步骤3】划分训练集和验证集')

    # 从训练集中划分验证集（取最后一次点击）
    trn_click_sorted = trn_click.sort_values('click_timestamp')
    trn_click_sorted['rank'] = trn_click_sorted.groupby('user_id')['click_timestamp'].rank(
        ascending=False, method='first'
    )

    # 验证集：每个用户最后一次点击
    val_click = trn_click_sorted[trn_click_sorted['rank'] == 1].copy()
    # 训练集历史：去除最后一次点击
    trn_hist_click = trn_click_sorted[trn_click_sorted['rank'] > 1].copy()

    logger.info(f'训练集历史点击数: {len(trn_hist_click)}')
    logger.info(f'验证集点击数: {len(val_click)}')

    # ==================== 4. 特征工程（优化：先负采样，后特征工程） ====================
    logger.info('\n【步骤4】特征工程')

    # 4.1 构建候选集（仅基础信息，不做特征工程）
    logger.info('构建候选集...')
    candidate_df = make_candidate_set(val_click, trn_user_recall_merged)

    # 4.2 负采样（在特征工程之前，大幅减少后续计算量）
    logger.info('负采样...')
    candidate_df = negative_sampling(candidate_df, neg_ratio=5, method='smart')
    logger.info(f'负采样后样本数: {len(candidate_df)}')

    # 4.3 为负采样后的候选集添加特征
    logger.info('构建特征（负采样后）...')
    train_df = add_features(candidate_df, trn_hist_click, all_click_df, item_info_df)

    # 保存特征
    train_df.to_pickle('user_data/features/train_features.pkl')
    logger.info('训练特征已保存')

    # 清理内存
    del trn_hist_click, val_click
    gc.collect()

    # ==================== 5. 排序模型训练 ====================
    logger.info('\n【步骤5】排序模型训练')

    # 5.1 LightGBM Ranker
    logger.info('\n--- 训练LightGBM Ranker ---')
    lgb_ranker = train_lgb_ranker(train_df)

    # 5.2 LightGBM Classifier
    logger.info('\n--- 训练LightGBM Classifier ---')
    lgb_classifier = train_lgb_classifier(train_df)

    # ==================== 6. 测试集预测 ====================
    logger.info('\n【步骤6】测试集预测')

    # 构建测试集特征
    logger.info('构建测试集特征...')
    test_df = make_test_set(tst_click, tst_user_recall_merged, all_click_df, item_info_df)

    # 保存
    test_df.to_pickle('user_data/features/test_features.pkl')

    # 预测
    logger.info('Ranker模型预测...')
    pred_ranker = predict_lgb(lgb_ranker, test_df)

    logger.info('Classifier模型预测...')
    pred_classifier = predict_lgb(lgb_classifier, test_df)

    # ==================== 7. 模型融合 ====================
    logger.info('\n【步骤7】模型融合')

    # 简单加权融合
    pred_final = pred_ranker.copy()
    pred_final['pred_score'] = 0.6 * pred_ranker['pred_score'] + 0.4 * pred_classifier['pred_score']

    # ==================== 8. 生成提交文件 ====================
    logger.info('\n【步骤8】生成提交文件')

    # 确保所有测试用户都有推荐
    test_user_ids = tst_click['user_id'].unique()
    pred_user_ids = pred_final['user_id'].unique()
    missing_users = set(test_user_ids) - set(pred_user_ids)

    if len(missing_users) > 0:
        logger.warning(f'发现{len(missing_users)}个用户没有预测结果，用热门文章填充')
        missing_df = []
        for user_id in missing_users:
            for rank, (item_id, score) in enumerate(hot_items[:5]):
                missing_df.append({
                    'user_id': user_id,
                    'click_article_id': item_id,
                    'pred_score': score
                })
        missing_df = pd.DataFrame(missing_df)
        pred_final = pd.concat([pred_final, missing_df], ignore_index=True)

    # 生成提交文件
    submit(pred_final, topk=5, model_name='final')

    # ==================== 完成 ====================
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info('\n' + '='*50)
    logger.info(f'实验完成！总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)')
    logger.info('='*50)


if __name__ == '__main__':
    main()
