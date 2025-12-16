"""
简化的Baseline脚本：ItemCF召回 + 直接提交

用于快速测试和获得初始结果
不需要特征工程和排序模型，直接使用召回分数生成提交
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle

# 导入自定义模块
from utils.data_loader import load_data, get_all_click_df, setup_logger
from recall.itemcf import itemcf_sim, itemcf_recall_batch, hot_recall, get_item_info_dict

# 设置日志
logger = setup_logger('baseline')


def generate_submission(user_recall_dict, hot_items, all_user_ids, save_path='prediction_result/baseline_result.csv'):
    """
    根据召回结果生成提交文件

    Args:
        user_recall_dict: {user_id: [(item_id, score), ...]}
        hot_items: 热门文章列表（用于填充）
        all_user_ids: 所有测试用户ID
        save_path: 保存路径

    Returns:
        submission: 提交DataFrame
    """
    logger.info('生成提交文件...')

    submissions = []

    for user_id in all_user_ids:
        if user_id in user_recall_dict and len(user_recall_dict[user_id]) >= 5:
            # 取Top 5
            top5_items = [item_id for item_id, _ in user_recall_dict[user_id][:5]]
        else:
            # 用热门文章填充
            if user_id in user_recall_dict:
                user_items = [item_id for item_id, _ in user_recall_dict[user_id]]
            else:
                user_items = []

            # 用热门文章补足5个
            hot_item_ids = [item_id for item_id, _ in hot_items]
            for item_id in hot_item_ids:
                if item_id not in user_items:
                    user_items.append(item_id)
                if len(user_items) >= 5:
                    break

            top5_items = user_items[:5]

        # 确保有5个推荐
        while len(top5_items) < 5:
            top5_items.append(hot_items[len(top5_items)][0])

        submissions.append({
            'user_id': user_id,
            'article_1': top5_items[0],
            'article_2': top5_items[1],
            'article_3': top5_items[2],
            'article_4': top5_items[3],
            'article_5': top5_items[4],
        })

    submission = pd.DataFrame(submissions)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)

    logger.info(f'提交文件已保存: {save_path}')
    logger.info(f'提交用户数: {len(submission)}')
    logger.info(f'前5行:\n{submission.head()}')

    return submission


def main_baseline():
    """Baseline主流程"""
    logger.info('='*50)
    logger.info('开始运行Baseline（ItemCF召回）')
    logger.info('='*50)

    start_time = datetime.now()

    # ==================== 1. 数据加载 ====================
    logger.info('\n【步骤1】数据加载')
    data_path = './Data/'

    trn_click, tst_click, item_info_df, item_emb_df = load_data(data_path)
    all_click_df = get_all_click_df(trn_click, tst_click, item_info_df)

    # 获取文章创建时间
    item_created_time_dict = get_item_info_dict(item_info_df)

    # 获取热门文章
    hot_items = hot_recall(all_click_df, top_k=50)

    # ==================== 2. ItemCF召回 ====================
    logger.info('\n【步骤2】ItemCF召回')

    itemcf_sim_path = 'user_data/itemcf_i2i_sim.pkl'
    if os.path.exists(itemcf_sim_path):
        logger.info('加载已有的ItemCF相似度矩阵...')
        with open(itemcf_sim_path, 'rb') as f:
            i2i_sim = pickle.load(f)
    else:
        logger.info('计算ItemCF相似度矩阵...')
        i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, itemcf_sim_path)

    # 测试集召回
    logger.info('测试集召回...')
    tst_user_recall = itemcf_recall_batch(tst_click, i2i_sim, top_k=10, item_num=50)

    # ==================== 3. 生成提交文件 ====================
    logger.info('\n【步骤3】生成提交文件')

    test_user_ids = tst_click['user_id'].unique()
    submission = generate_submission(tst_user_recall, hot_items, test_user_ids,
                                    save_path='prediction_result/baseline_result.csv')

    # ==================== 完成 ====================
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info('\n' + '='*50)
    logger.info(f'Baseline完成！总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)')
    logger.info('='*50)


if __name__ == '__main__':
    main_baseline()
