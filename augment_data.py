# -*- coding: utf-8 -*-
"""
数据增强脚本
1. 用户行为序列补全: 基于相似用户的行为迁移
2. 半监督学习: 利用商品语义相似性生成伪标签
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm


def augment_by_similar_users(history_path, output_path, text_feature_path,
                             min_interactions=8, top_k_neighbors=5, max_augment_per_user=3):
    """
    基于相似用户的行为迁移补全
    对于交互数不足的用户，从相似用户的历史中补充商品
    """
    history = pd.read_csv(history_path)
    history["item_id"] = history["item_id"].astype(str)

    text_features = np.load(text_feature_path, allow_pickle=True).item()

    user_items = defaultdict(list)
    for _, row in history.iterrows():
        user_items[row["user_id"]].append(str(row["item_id"]))

    user_profiles = {}
    for uid, items in user_items.items():
        vecs = [text_features[iid] for iid in items if iid in text_features]
        if vecs:
            user_profiles[uid] = np.mean(vecs, axis=0)

    user_ids = list(user_profiles.keys())
    if len(user_ids) < 2:
        print("Not enough users for augmentation")
        return

    profile_matrix = np.array([user_profiles[uid] for uid in user_ids])
    profile_matrix = profile_matrix / np.maximum(
        np.linalg.norm(profile_matrix, axis=1, keepdims=True), 1e-9
    )

    sim_matrix = cosine_similarity(profile_matrix)

    new_rows = []
    augmented_users = 0

    for i, uid in enumerate(tqdm(user_ids, desc="Augmenting users")):
        current_items = set(user_items[uid])
        if len(current_items) >= min_interactions:
            continue

        neighbor_scores = sim_matrix[i].copy()
        neighbor_scores[i] = -1
        top_neighbors = np.argsort(neighbor_scores)[-top_k_neighbors:]

        added = 0
        for ni in top_neighbors:
            neighbor_uid = user_ids[ni]
            neighbor_items = user_items[neighbor_uid]

            for iid in neighbor_items:
                if iid not in current_items and iid in text_features:
                    max_ts = history[history["user_id"] == uid]["timestamp"].max()
                    new_rows.append({
                        "user_id": uid,
                        "item_id": iid,
                        "timestamp": max_ts + added + 1,
                        "source": "augmented"
                    })
                    current_items.add(iid)
                    added += 1
                    if added >= max_augment_per_user:
                        break
            if added >= max_augment_per_user:
                break

        if added > 0:
            augmented_users += 1

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        combined = pd.concat([history, aug_df], ignore_index=True)
        combined = combined.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        print(f"Augmented {augmented_users} users with {len(new_rows)} new interactions")
        print(f"Original: {len(history)} -> Augmented: {len(combined)}")
        print(f"Saved to: {output_path}")
    else:
        print("No augmentation needed")

    return len(new_rows)


def augment_by_semantic_similarity(history_path, items_path, output_path,
                                   text_feature_path, top_k_similar=5):
    """
    半监督数据增强: 利用商品语义相似性生成伪交互
    对于用户交互过的每个商品，将语义最相似的商品也加入交互历史
    """
    history = pd.read_csv(history_path)
    history["item_id"] = history["item_id"].astype(str)

    items = pd.read_csv(items_path)
    items["item_id"] = items["item_id"].astype(str)

    text_features = np.load(text_feature_path, allow_pickle=True).item()

    item_ids = list(text_features.keys())
    item_matrix = np.array([text_features[iid] for iid in item_ids])
    item_matrix = item_matrix / np.maximum(
        np.linalg.norm(item_matrix, axis=1, keepdims=True), 1e-9
    )

    user_items = defaultdict(set)
    for _, row in history.iterrows():
        user_items[row["user_id"]].add(str(row["item_id"]))

    new_rows = []
    augmented_users = 0

    for uid, item_set_orig in tqdm(user_items.items(), desc="Semantic augmentation"):
        item_set = set(item_set_orig)
        original_items = list(item_set_orig)
        added = 0
        for iid in original_items:
            if iid not in text_features:
                continue
            idx = item_ids.index(iid)
            sims = cosine_similarity([item_matrix[idx]], item_matrix)[0]
            top_indices = np.argsort(sims)[-top_k_similar - 1:-1]

            for ti in top_indices:
                sim_iid = item_ids[ti]
                if sim_iid not in item_set:
                    max_ts = history[history["user_id"] == uid]["timestamp"].max()
                    new_rows.append({
                        "user_id": uid,
                        "item_id": sim_iid,
                        "timestamp": max_ts + added + 1,
                        "source": "semantic_augmented"
                    })
                    item_set.add(sim_iid)
                    added += 1
                    if added >= 5:
                        break
            if added >= 5:
                break

        if added > 0:
            augmented_users += 1

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        combined = pd.concat([history, aug_df], ignore_index=True)
        combined = combined.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        print(f"Semantic augmented {augmented_users} users with {len(new_rows)} interactions")
        print(f"Original: {len(history)} -> Augmented: {len(combined)}")
    else:
        print("No semantic augmentation needed")

    return len(new_rows)


if __name__ == "__main__":
    from data_config import config

    print("=" * 60)
    print("数据增强 - 用户行为序列补全")
    print("=" * 60)

    aug_path = config.user_history_csv.replace(".csv", "_augmented.csv")

    n1 = augment_by_similar_users(
        history_path=config.user_history_csv,
        output_path=aug_path,
        text_feature_path=config.text_features,
        min_interactions=8,
        top_k_neighbors=5,
        max_augment_per_user=3
    )

    print("\n" + "=" * 60)
    print("数据增强 - 语义相似性伪标签")
    print("=" * 60)

    aug_path2 = config.user_history_csv.replace(".csv", "_augmented_full.csv")
    n2 = augment_by_semantic_similarity(
        history_path=aug_path if os.path.exists(aug_path) else config.user_history_csv,
        items_path=config.items_csv,
        output_path=aug_path2,
        text_feature_path=config.text_features,
        top_k_similar=3
    )

    print(f"\nTotal augmented interactions: {n1 + n2}")
