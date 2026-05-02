# -*- coding: utf-8 -*-
"""
LambdaMART训练脚本
使用LightGBM的LambdaMART目标函数训练排序模型
"""

import os
import json
import pickle
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
os.environ["HF_HOME"] = "D:\\HF_HOME"
os.environ["TRANSFORMERS_CACHE"] = "D:\\HF_HOME\\transformers"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from data_config import config
from lambdamart_features import LambdaMARTFeatureExtractor
from recall_faiss_v2 import FaissRecallV2
from user_model import UserInterestModel


def build_leave_one_out_split(history_path, holdout_len=2, max_users=None):
    """
    构建Leave-One-Out训练/测试划分
    """
    history = pd.read_csv(history_path).sort_values(["user_id", "timestamp"])
    history["item_id"] = history["item_id"].astype(str)

    if max_users is not None and max_users > 0:
        keep_users = list(dict.fromkeys(history["user_id"].tolist()))[:max_users]
        history = history[history["user_id"].isin(keep_users)].copy()

    train_rows = []
    ground_truth = {}

    for user_id, user_df in history.groupby("user_id"):
        user_rows = user_df.to_dict("records")
        if len(user_rows) <= holdout_len:
            continue
        ground_truth[user_id] = [str(row["item_id"]) for row in user_rows[-holdout_len:]]
        train_rows.extend(user_rows[:-holdout_len])

    train_history = pd.DataFrame(train_rows)
    return train_history, ground_truth


def generate_candidates_for_users(faiss_recall, train_history_items, ground_truth, topk=100):
    """
    为每个用户生成召回候选
    """
    candidates_dict = {}

    for user_id in ground_truth:
        history_items = train_history_items.get(user_id, [])

        candidates = faiss_recall.recall_by_user(user_id, topk=topk)

        history_set = set(history_items)
        filtered = [(cid, score) for cid, score in candidates if cid not in history_set]

        candidates_dict[user_id] = filtered

    return candidates_dict


def train_lambdamart_model(X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names):
    """
    训练LambdaMART模型
    """
    print("\n训练LambdaMART模型...")
    print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
    print(f"特征数量: {len(feature_names)}")

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
        feature_name=feature_names
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=groups_val,
        feature_name=feature_names,
        reference=train_data
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,
        "ndcg_eval_at": [5, 10],
        "label_gain": [0, 1],
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=20)
        ]
    )

    return model


def main():
    print("=" * 60)
    print("LambdaMART训练脚本")
    print("=" * 60)

    max_users = 2000
    topk_recall = 100

    print(f"\n[1] 加载数据和构建索引...")

    train_history, ground_truth = build_leave_one_out_split(
        config.user_history_csv,
        holdout_len=2,
        max_users=max_users
    )

    print(f"用户数: {len(ground_truth)}")
    print(f"训练交互数: {len(train_history)}")

    train_history_items = {}
    for _, row in train_history.iterrows():
        uid = row["user_id"]
        iid = str(row["item_id"])
        train_history_items.setdefault(uid, []).append(iid)

    print(f"\n[2] 初始化FAISS召回...")

    user_model = UserInterestModel(
        history_path=config.user_history_csv,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    )

    faiss_recall = FaissRecallV2(
        user_model=user_model,
        text_index_path=config.text_index,
        text_ids_path=config.text_ids,
        image_index_path=config.image_index,
        image_ids_path=config.image_ids
    )

    print(f"\n[3] 生成召回候选...")

    candidates_dict = generate_candidates_for_users(
        faiss_recall, train_history_items, ground_truth, topk=topk_recall
    )

    print(f"候选生成完成: {len(candidates_dict)} 用户")

    print(f"\n[4] 提取LambdaMART特征...")

    extractor = LambdaMARTFeatureExtractor(
        text_features_path=config.text_features,
        image_features_path=config.image_features,
        multimodal_features_path=config.multimodal_features,
        history_path=config.user_history_csv
    )

    X, y, groups, feature_names = extractor.extract_training_data(
        train_history, ground_truth, candidates_dict, topk=topk_recall
    )

    print(f"总样本数: {len(X)}")
    print(f"正样本数: {sum(y > 0)}")
    print(f"负样本数: {sum(y == 0)}")
    print(f"特征数: {len(feature_names)}")

    print(f"\n[5] 划分训练/验证集...")

    unique_groups = []
    for i, g in enumerate(groups):
        unique_groups.extend([i] * g)

    unique_groups = np.array(unique_groups)

    group_ids = np.unique(unique_groups)
    train_group_ids, val_group_ids = train_test_split(
        group_ids, test_size=0.2, random_state=42
    )

    train_mask = np.isin(unique_groups, train_group_ids)
    val_mask = np.isin(unique_groups, val_group_ids)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    groups_train = [groups[i] for i in train_group_ids]
    groups_val = [groups[i] for i in val_group_ids]

    print(f"训练集: {len(X_train)} 样本, {len(groups_train)} 用户")
    print(f"验证集: {len(X_val)} 样本, {len(groups_val)} 用户")

    print(f"\n[6] 训练LambdaMART模型...")

    model = train_lambdamart_model(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val,
        feature_names
    )

    print(f"\n[7] 保存模型...")

    model_path = os.path.join("models", "lambdamart_ranker.txt")
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    print(f"模型已保存: {model_path}")

    model_path_pkl = os.path.join("models", "lambdamart_ranker.pkl")
    with open(model_path_pkl, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "extractor": extractor
        }, f)
    print(f"模型(含特征提取器)已保存: {model_path_pkl}")

    print(f"\n[8] 特征重要性...")

    importance = model.feature_importance(importance_type="gain")
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    print("\nTop 10 重要特征:")
    for name, imp in feature_importance[:10]:
        print(f"  {name}: {imp:.2f}")

    print("\n" + "=" * 60)
    print("LambdaMART训练完成!")
    print("=" * 60)

    return model, feature_names, extractor


if __name__ == "__main__":
    main()
