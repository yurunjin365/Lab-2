"""
特征工程：构建用户-文章特征

特征类别：
1. 用户画像特征：活跃度、设备习惯、时间习惯、主题偏好、字数偏好
2. 文章属性特征：热度、类别、字数、创建时间
3. 交叉特征：用户历史行为与候选文章的相似度、时间差、字数差等
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger()


def create_user_features(all_click_df):
    """
    创建用户画像特征

    Args:
        all_click_df: 所有点击数据

    Returns:
        user_features: 用户特征DataFrame
    """
    logger.info('开始构建用户特征...')

    user_features = pd.DataFrame()

    # 1. 用户活跃度特征
    logger.info('构建用户活跃度特征...')
    user_click_count = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    user_click_count.columns = ['user_id', 'user_click_count']
    user_features = user_click_count

    # 用户点击时间跨度
    user_time_span = all_click_df.groupby('user_id')['click_timestamp'].agg(['min', 'max']).reset_index()
    user_time_span['user_time_span'] = user_time_span['max'] - user_time_span['min']
    user_features = user_features.merge(user_time_span[['user_id', 'user_time_span']], on='user_id', how='left')

    # 用户平均点击间隔
    user_features['user_avg_click_interval'] = user_features['user_time_span'] / (user_features['user_click_count'] + 1)

    # 2. 设备习惯特征（众数）
    logger.info('构建用户设备习惯特征...')
    device_cols = ['click_environment', 'click_deviceGroup', 'click_os',
                   'click_country', 'click_region', 'click_referrer_type']

    for col in device_cols:
        user_device = all_click_df.groupby('user_id')[col].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else -1).reset_index()
        user_device.columns = ['user_id', f'user_{col}_mode']
        user_features = user_features.merge(user_device, on='user_id', how='left')

    # 3. 主题偏好特征
    logger.info('构建用户主题偏好特征...')
    # 用户点击过的文章类别数
    user_category_count = all_click_df.groupby('user_id')['category_id'].nunique().reset_index()
    user_category_count.columns = ['user_id', 'user_category_count']
    user_features = user_features.merge(user_category_count, on='user_id', how='left')

    # 用户最常看的类别（众数）
    user_category_mode = all_click_df.groupby('user_id')['category_id'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else -1).reset_index()
    user_category_mode.columns = ['user_id', 'user_category_mode']
    user_features = user_features.merge(user_category_mode, on='user_id', how='left')

    # 4. 字数偏好特征
    logger.info('构建用户字数偏好特征...')
    user_words = all_click_df.groupby('user_id')['words_count'].agg(['mean', 'std', 'min', 'max']).reset_index()
    user_words.columns = ['user_id', 'user_words_mean', 'user_words_std', 'user_words_min', 'user_words_max']
    user_features = user_features.merge(user_words, on='user_id', how='left')

    logger.info(f'用户特征构建完成，特征数: {user_features.shape[1] - 1}')
    return user_features


def create_item_features(all_click_df, item_info_df):
    """
    创建文章属性特征

    Args:
        all_click_df: 所有点击数据
        item_info_df: 文章信息

    Returns:
        item_features: 文章特征DataFrame
    """
    logger.info('开始构建文章特征...')

    # 基础属性
    item_features = item_info_df.copy()

    # 1. 文章热度特征
    logger.info('构建文章热度特征...')
    item_click_count = all_click_df.groupby('click_article_id')['user_id'].count().reset_index()
    item_click_count.columns = ['click_article_id', 'item_click_count']
    item_features = item_features.merge(item_click_count, on='click_article_id', how='left')
    item_features['item_click_count'].fillna(0, inplace=True)

    # 文章被多少不同用户点击
    item_user_count = all_click_df.groupby('click_article_id')['user_id'].nunique().reset_index()
    item_user_count.columns = ['click_article_id', 'item_user_count']
    item_features = item_features.merge(item_user_count, on='click_article_id', how='left')
    item_features['item_user_count'].fillna(0, inplace=True)

    # 2. 文章时效性特征
    # 文章首次被点击的时间
    item_first_click = all_click_df.groupby('click_article_id')['click_timestamp'].min().reset_index()
    item_first_click.columns = ['click_article_id', 'item_first_click_time']
    item_features = item_features.merge(item_first_click, on='click_article_id', how='left')

    # 文章最后一次被点击的时间
    item_last_click = all_click_df.groupby('click_article_id')['click_timestamp'].max().reset_index()
    item_last_click.columns = ['click_article_id', 'item_last_click_time']
    item_features = item_features.merge(item_last_click, on='click_article_id', how='left')

    logger.info(f'文章特征构建完成，特征数: {item_features.shape[1] - 1}')
    return item_features


def create_cross_features(user_hist_df, candidate_df, user_features, item_features):
    """
    创建用户-文章交叉特征

    Args:
        user_hist_df: 用户历史点击数据
        candidate_df: 候选文章数据 (user_id, click_article_id)
        user_features: 用户特征
        item_features: 文章特征

    Returns:
        cross_features: 交叉特征DataFrame
    """
    logger.info('开始构建交叉特征...')

    # 合并用户和文章特征
    df = candidate_df.copy()
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='click_article_id', how='left')

    # 构建用户历史点击信息
    user_hist_dict = user_hist_df.groupby('user_id').agg({
        'click_article_id': list,
        'category_id': list,
        'words_count': list,
        'click_timestamp': list,
        'created_at_ts': list
    }).to_dict('index')

    # 1. 候选文章与用户历史的相似度特征
    logger.info('构建相似度特征...')

    def compute_sim_features(row):
        user_id = row['user_id']
        cand_category = row['category_id']
        cand_words = row['words_count']
        cand_created = row['created_at_ts']

        if user_id not in user_hist_dict:
            return pd.Series({
                'cand_in_user_category': 0,
                'user_cand_category_match_rate': 0,
                'user_cand_words_diff_mean': 0,
                'user_cand_words_diff_std': 0,
                'user_cand_created_time_diff_mean': 0,
            })

        hist_info = user_hist_dict[user_id]
        hist_categories = hist_info['category_id']
        hist_words = hist_info['words_count']
        hist_created = hist_info['created_at_ts']

        # 候选文章类别是否在用户历史中
        cand_in_user_category = 1 if cand_category in hist_categories else 0

        # 用户历史中该类别占比
        user_cand_category_match_rate = hist_categories.count(cand_category) / len(hist_categories) if len(hist_categories) > 0 else 0

        # 字数差异
        words_diffs = [abs(cand_words - w) for w in hist_words]
        user_cand_words_diff_mean = np.mean(words_diffs) if len(words_diffs) > 0 else 0
        user_cand_words_diff_std = np.std(words_diffs) if len(words_diffs) > 0 else 0

        # 创建时间差异
        created_diffs = [abs(cand_created - c) for c in hist_created]
        user_cand_created_time_diff_mean = np.mean(created_diffs) if len(created_diffs) > 0 else 0

        return pd.Series({
            'cand_in_user_category': cand_in_user_category,
            'user_cand_category_match_rate': user_cand_category_match_rate,
            'user_cand_words_diff_mean': user_cand_words_diff_mean,
            'user_cand_words_diff_std': user_cand_words_diff_std,
            'user_cand_created_time_diff_mean': user_cand_created_time_diff_mean,
        })

    sim_features = df[['user_id', 'click_article_id', 'category_id', 'words_count', 'created_at_ts']].progress_apply(
        compute_sim_features, axis=1
    )

    df = pd.concat([df, sim_features], axis=1)

    logger.info(f'交叉特征构建完成，总特征数: {df.shape[1]}')
    return df


def make_train_set(trn_click, val_click, user_recall_dict, all_click_df, item_info_df,
                   is_val=True):
    """
    构建训练集/验证集

    Args:
        trn_click: 训练集历史点击（用于构建特征）
        val_click: 验证集点击（作为标签）
        user_recall_dict: 召回结果 {user_id: [(item_id, score), ...]}
        all_click_df: 所有点击数据（用于构建特征）
        item_info_df: 文章信息
        is_val: 是否是验证集

    Returns:
        train_df: 训练集DataFrame，包含label列
    """
    logger.info(f'开始构建{"验证" if is_val else "训练"}集...')

    # 构建候选集
    candidates = []
    labels = []

    val_labels = dict(zip(val_click['user_id'], val_click['click_article_id']))

    for user_id, recall_items in tqdm(user_recall_dict.items(), desc='构建候选集'):
        if user_id not in val_labels:
            continue

        true_item = val_labels[user_id]

        for item_id, score in recall_items:
            candidates.append({
                'user_id': user_id,
                'click_article_id': item_id,
                'recall_score': score
            })
            # 标签：1表示真实点击，0表示未点击
            labels.append(1 if item_id == true_item else 0)

    candidate_df = pd.DataFrame(candidates)
    candidate_df['label'] = labels

    logger.info(f'候选集大小: {len(candidate_df)}')
    logger.info(f'正样本数: {sum(labels)}, 负样本数: {len(labels) - sum(labels)}')

    # 合并文章信息
    candidate_df = candidate_df.merge(item_info_df, on='click_article_id', how='left')

    # 构建用户和文章特征
    user_features = create_user_features(all_click_df)
    item_features = create_item_features(all_click_df, item_info_df)

    # 构建交叉特征
    tqdm.pandas(desc='构建交叉特征')
    train_df = create_cross_features(trn_click, candidate_df, user_features, item_features)

    logger.info(f'{"验证" if is_val else "训练"}集构建完成，shape: {train_df.shape}')

    return train_df


def make_test_set(tst_click, user_recall_dict, all_click_df, item_info_df):
    """
    构建测试集

    Args:
        tst_click: 测试集点击（用于构建特征）
        user_recall_dict: 召回结果
        all_click_df: 所有点击数据
        item_info_df: 文章信息

    Returns:
        test_df: 测试集DataFrame
    """
    logger.info('开始构建测试集...')

    # 构建候选集
    candidates = []

    for user_id, recall_items in tqdm(user_recall_dict.items(), desc='构建测试候选集'):
        for item_id, score in recall_items:
            candidates.append({
                'user_id': user_id,
                'click_article_id': item_id,
                'recall_score': score
            })

    candidate_df = pd.DataFrame(candidates)
    logger.info(f'测试候选集大小: {len(candidate_df)}')

    # 合并文章信息
    candidate_df = candidate_df.merge(item_info_df, on='click_article_id', how='left')

    # 构建用户和文章特征
    user_features = create_user_features(all_click_df)
    item_features = create_item_features(all_click_df, item_info_df)

    # 构建交叉特征
    tqdm.pandas(desc='构建交叉特征')
    test_df = create_cross_features(tst_click, candidate_df, user_features, item_features)

    logger.info(f'测试集构建完成，shape: {test_df.shape}')

    return test_df
