"""
LightGBM排序模型

包含两种模型：
1. LightGBM Ranker：学习排序任务，使用lambdarank目标函数
2. LightGBM Classifier：二分类任务，预测点击概率

支持GPU加速（需要安装lightgbm GPU版本）
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import logging
import pickle
import os

logger = logging.getLogger()

# 检测GPU是否可用
def check_gpu_available():
    """检测LightGBM GPU是否可用"""
    try:
        # 尝试创建一个小的GPU数据集来测试
        import tempfile
        test_data = lgb.Dataset(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0, 'verbose': -1}
        with tempfile.TemporaryDirectory() as tmpdir:
            lgb.train(params, test_data, num_boost_round=1, verbose_eval=False)
        return True
    except Exception as e:
        logger.warning(f'LightGBM GPU不可用: {e}')
        return False

# 全局GPU可用标志（只检测一次）
_GPU_AVAILABLE = None

def is_gpu_available():
    """获取GPU可用状态（缓存结果）"""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        logger.info('检测LightGBM GPU支持...')
        _GPU_AVAILABLE = check_gpu_available()
        if _GPU_AVAILABLE:
            logger.info('✅ LightGBM GPU加速已启用')
        else:
            logger.info('⚠️ LightGBM GPU不可用，使用CPU训练')
    return _GPU_AVAILABLE


def get_lgb_features(df):
    """
    获取LightGBM特征列

    Args:
        df: DataFrame

    Returns:
        feature_cols: 特征列名列表
    """
    # 排除的列
    exclude_cols = ['user_id', 'click_article_id', 'label']

    # 特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f'LightGBM特征数: {len(feature_cols)}')
    logger.info(f'特征列: {feature_cols[:20]}...')

    return feature_cols


def train_lgb_ranker(train_df, val_df=None, params=None):
    """
    训练LightGBM Ranker模型

    Args:
        train_df: 训练集，需包含user_id, label列
        val_df: 验证集（可选，若不提供则自动划分20%作为验证）
        params: 模型参数（可选）

    Returns:
        model: 训练好的模型
    """
    logger.info('开始训练LightGBM Ranker模型...')

    # 默认参数（优化版：防止过拟合，加速训练）
    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,  # 提高学习率，加速收敛
            'feature_fraction': 0.8,  # 降低，增加随机性
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,  # 减少冗余输出
            'min_child_samples': 50,  # 提高，防止过拟合
            'lambda_l1': 0.1,  # L1正则化
            'lambda_l2': 0.1,  # L2正则化
            'min_gain_to_split': 0.01,  # 最小分裂增益阈值
        }
        
        # 尝试启用GPU加速
        if is_gpu_available():
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            logger.info('Ranker模型使用GPU训练')

    # 特征列
    feature_cols = get_lgb_features(train_df)

    # 如果没有验证集，从训练集划分20%作为验证集（按用户划分）
    if val_df is None:
        logger.info('未提供验证集，自动划分20%用户作为验证集...')
        unique_users = train_df['user_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_users)
        val_user_count = int(len(unique_users) * 0.2)
        val_users = set(unique_users[:val_user_count])
        
        val_df = train_df[train_df['user_id'].isin(val_users)].copy()
        train_df = train_df[~train_df['user_id'].isin(val_users)].copy()
        logger.info(f'训练集用户数: {len(unique_users) - val_user_count}, 验证集用户数: {val_user_count}')

    # 准备训练数据
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    group_train = train_df.groupby('user_id').size().values

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)

    # 准备验证数据
    X_val = val_df[feature_cols]
    y_val = val_df['label']
    group_val = val_df.groupby('user_id').size().values
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

    valid_sets = [train_data, val_data]
    valid_names = ['train', 'valid']

    # 用于记录训练过程的指标
    evals_result = {}
    
    # 自定义回调函数：每轮输出详细的训练指标到logger
    def log_evaluation_to_logger(period=10):
        def callback(env):
            if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
                result_str = f'Epoch: {env.iteration}'
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    result_str += f', {data_name}_{eval_name}: {result:.6f}'
                logger.info(result_str)
        return callback
    
    # 训练模型
    logger.info('开始训练...')
    logger.info(f'Learning Rate: {params.get("learning_rate", 0.1)}')
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.record_evaluation(evals_result),
            log_evaluation_to_logger(period=1)  # 每轮都输出到logger
        ]
    )

    logger.info('LightGBM Ranker训练完成')
    
    # 记录训练loss/metric摘要到日志
    if evals_result:
        logger.info('=== Ranker训练指标摘要 ===')
        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    logger.info(f'{dataset_name} - {metric_name}: 初始={values[0]:.6f}, 最终={values[-1]:.6f}, 最佳={max(values):.6f}')
        # 保存完整训练历史
        import json
        history_path = 'user_data/model_data/lgb_ranker_history.json'
        with open(history_path, 'w') as f:
            json.dump(evals_result, f, indent=2)
        logger.info(f'训练历史已保存: {history_path}')

    # 保存模型
    model_path = 'user_data/model_data/lgb_ranker.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'模型已保存: {model_path}')

    return model


def train_lgb_classifier(train_df, val_df=None, params=None):
    """
    训练LightGBM Classifier模型（二分类）

    Args:
        train_df: 训练集，需包含label列
        val_df: 验证集（可选，若不提供则自动划分20%作为验证）
        params: 模型参数（可选）

    Returns:
        model: 训练好的模型
    """
    logger.info('开始训练LightGBM Classifier模型...')

    # 默认参数（优化版：防止过拟合，加速训练）
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,  # 提高学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,  # 减少冗余输出
            'min_child_samples': 50,  # 防止过拟合
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.01,
        }
        
        # 尝试启用GPU加速
        if is_gpu_available():
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            logger.info('Classifier模型使用GPU训练')

    # 特征列
    feature_cols = get_lgb_features(train_df)

    # 如果没有验证集，从训练集划分20%作为验证集
    if val_df is None:
        logger.info('未提供验证集，自动划分20%数据作为验证集...')
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
        logger.info(f'训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}')

    # 准备训练数据
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    train_data = lgb.Dataset(X_train, label=y_train)

    # 准备验证数据
    X_val = val_df[feature_cols]
    y_val = val_df['label']
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    valid_sets = [train_data, val_data]
    valid_names = ['train', 'valid']

    # 用于记录训练过程的指标
    evals_result = {}
    
    # 自定义回调函数：每轮输出详细的训练指标到logger
    def log_evaluation_to_logger(period=10):
        def callback(env):
            if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
                result_str = f'Epoch: {env.iteration}'
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    result_str += f', {data_name}_{eval_name}: {result:.6f}'
                logger.info(result_str)
        return callback
    
    # 训练模型
    logger.info('开始训练...')
    logger.info(f'Learning Rate: {params.get("learning_rate", 0.1)}')
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.record_evaluation(evals_result),
            log_evaluation_to_logger(period=1)  # 每轮都输出到logger
        ]
    )

    logger.info('LightGBM Classifier训练完成')
    
    # 记录训练loss/metric摘要到日志
    if evals_result:
        logger.info('=== Classifier训练指标摘要 ===')
        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    logger.info(f'{dataset_name} - {metric_name}: 初始={values[0]:.6f}, 最终={values[-1]:.6f}, 最佳={max(values):.6f}')
        # 保存完整训练历史
        import json
        history_path = 'user_data/model_data/lgb_classifier_history.json'
        with open(history_path, 'w') as f:
            json.dump(evals_result, f, indent=2)
        logger.info(f'训练历史已保存: {history_path}')

    # 保存模型
    model_path = 'user_data/model_data/lgb_classifier.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'模型已保存: {model_path}')

    return model


def predict_lgb(model, test_df):
    """
    使用LightGBM模型预测

    Args:
        model: 训练好的模型
        test_df: 测试集

    Returns:
        pred_df: 包含预测分数的DataFrame
    """
    logger.info('开始预测...')

    feature_cols = get_lgb_features(test_df)
    X_test = test_df[feature_cols]

    # 预测
    preds = model.predict(X_test)

    # 构建结果
    pred_df = test_df[['user_id', 'click_article_id']].copy()
    pred_df['pred_score'] = preds

    logger.info(f'预测完成，结果数量: {len(pred_df)}')

    return pred_df


def negative_sampling(train_df, neg_ratio=5, method='random'):
    """
    负采样：从负样本中采样

    Args:
        train_df: 训练集
        neg_ratio: 负样本采样比例（相对于正样本）
        method: 采样方法 'random' 或 'smart'

    Returns:
        sampled_df: 采样后的训练集
    """
    logger.info(f'开始负采样，负样本比例: {neg_ratio}...')

    pos_df = train_df[train_df['label'] == 1]
    neg_df = train_df[train_df['label'] == 0]

    logger.info(f'原始正样本数: {len(pos_df)}, 负样本数: {len(neg_df)}')

    # 计算需要采样的负样本数
    neg_sample_num = len(pos_df) * neg_ratio

    if method == 'random':
        # 随机采样
        neg_sampled = neg_df.sample(n=min(neg_sample_num, len(neg_df)), random_state=2021)
    elif method == 'smart':
        # 智能采样：保证每个用户和每篇文章都有负样本
        # 按用户采样一部分
        neg_by_user = neg_df.groupby('user_id').sample(n=1, random_state=2021)
        # 按文章采样一部分
        neg_by_item = neg_df.groupby('click_article_id').sample(n=1, random_state=2021)
        # 随机采样剩余部分
        neg_used = set(neg_by_user.index) | set(neg_by_item.index)
        neg_remain = neg_df[~neg_df.index.isin(neg_used)]
        neg_random = neg_remain.sample(n=min(neg_sample_num - len(neg_used), len(neg_remain)),
                                      random_state=2021)
        neg_sampled = pd.concat([neg_by_user, neg_by_item, neg_random]).drop_duplicates()
    else:
        raise ValueError(f'Unknown method: {method}')

    # 合并
    sampled_df = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=2021).reset_index(drop=True)

    logger.info(f'采样后正样本数: {sum(sampled_df["label"] == 1)}, 负样本数: {sum(sampled_df["label"] == 0)}')

    return sampled_df
