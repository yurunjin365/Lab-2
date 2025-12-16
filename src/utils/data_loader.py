"""
数据加载与内存优化工具
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime

# 配置日志
def setup_logger(log_name='experiment'):
    """设置带时间戳的日志"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/{log_name}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logger()


def reduce_mem(df, cols=None):
    """
    内存优化函数：降低DataFrame的内存占用

    Args:
        df: pandas DataFrame
        cols: 需要优化的列名列表，None表示所有列

    Returns:
        优化后的DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info(f'原始内存占用: {start_mem:.2f} MB')

    if cols is None:
        cols = df.columns.tolist()

    for col in cols:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info(f'优化后内存占用: {end_mem:.2f} MB')
    logger.info(f'内存减少: {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


def load_data(data_path='./Data/', mode='offline'):
    """
    加载数据文件

    Args:
        data_path: 数据目录路径
        mode: 'offline'离线验证模式 或 'online'在线提交模式

    Returns:
        trn_click: 训练集点击日志
        tst_click: 测试集点击日志
        item_info_df: 文章信息
        item_emb_df: 文章embedding
    """
    logger.info(f'开始加载数据，模式: {mode}')
    start_time = datetime.now()

    # 加载训练集点击日志
    logger.info('加载训练集点击日志...')
    trn_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
    trn_click = reduce_mem(trn_click)

    # 加载测试集点击日志
    logger.info('加载测试集点击日志...')
    tst_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))
    tst_click = reduce_mem(tst_click)

    # 加载文章信息
    logger.info('加载文章信息...')
    item_info_df = pd.read_csv(os.path.join(data_path, 'articles.csv'))
    item_info_df = reduce_mem(item_info_df)
    item_info_df.rename(columns={'article_id': 'click_article_id'}, inplace=True)

    # 加载文章embedding
    logger.info('加载文章embedding...')
    item_emb_df = pd.read_csv(os.path.join(data_path, 'articles_emb.csv'))
    item_emb_df = reduce_mem(item_emb_df)

    # 数据基本统计
    logger.info(f'训练集用户数: {trn_click.user_id.nunique()}')
    logger.info(f'训练集点击记录数: {len(trn_click)}')
    logger.info(f'测试集用户数: {tst_click.user_id.nunique()}')
    logger.info(f'测试集点击记录数: {len(tst_click)}')
    logger.info(f'文章总数: {len(item_info_df)}')
    logger.info(f'文章embedding数: {len(item_emb_df)}')

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f'数据加载完成，耗时: {elapsed:.2f}秒')

    return trn_click, tst_click, item_info_df, item_emb_df


def save_pickle(data, file_path):
    """保存pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f'已保存: {file_path}')


def load_pickle(file_path):
    """加载pickle文件"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'已加载: {file_path}')
    return data


def get_all_click_df(trn_click, tst_click, item_info_df):
    """
    合并训练集和测试集的点击数据（用于计算全局统计信息）

    Args:
        trn_click: 训练集点击日志
        tst_click: 测试集点击日志
        item_info_df: 文章信息

    Returns:
        all_click: 合并后的完整点击数据
    """
    logger.info('合并训练集和测试集点击数据...')
    all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    # 合并文章信息
    all_click = all_click.merge(item_info_df, on='click_article_id', how='left')

    # 排序
    all_click = all_click.sort_values('click_timestamp')

    logger.info(f'合并后总点击记录数: {len(all_click)}')
    logger.info(f'合并后总用户数: {all_click.user_id.nunique()}')
    logger.info(f'合并后总文章数: {all_click.click_article_id.nunique()}')

    return all_click
