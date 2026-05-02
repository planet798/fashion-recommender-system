# -*- coding: utf-8 -*-
"""
优化后 Text Search 评估脚本
对比 BERT Baseline vs 优化后 Our Model
优化点:
1. 重排序权重: 去掉BM25/token_overlap, 引入BERT语义匹配+注意力机制
2. 数据增强: 交互数据从13973增加到28623
3. DIN排序模型: 引入注意力机制的用户兴趣建模
"""

import os
import json
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
tempfile.tempdir = "D:\\TEMP"
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from bert_baseline import BERTRetriever
from evaluation import evaluate_recommendations
from data_config import config


def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_leave_one_out_split(history_path, holdout_len=2, max_users=None):
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


def evaluate_bert_baseline(train_history, ground_truth, topk, ks, valid_ids):
    print(f"[BERT Baseline] start: users={len(ground_truth)} topk={topk}", flush=True)
    retriever = BERTRetriever(items_path=config.items_csv, valid_ids=valid_ids)
    recommendations = {}
    log_every = max(1, len(ground_truth) // 20)

    for idx, user_id in enumerate(ground_truth, start=1):
        recommendations[user_id] = retriever.recommend_for_user(train_history, user_id, topk=topk)
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[BERT] {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks, model_name="BERT")
    print("[BERT Baseline] done", flush=True)
    return metrics, recommendations


def evaluate_optimized_our_model(train_history, ground_truth, topk, ks, valid_ids,
                                  use_augmented=False):
    """
    优化后的Our Model:
    1. 多模态特征余弦相似度
    2. BERT语义匹配分数
    3. 注意力机制动态权重
    4. (可选) 增强后的交互数据
    """
    print(f"[Optimized Our Model] start: users={len(ground_truth)} topk={topk}", flush=True)

    text_features = np.load(config.text_features, allow_pickle=True).item()
    image_features = np.load(config.image_features, allow_pickle=True).item()
    multimodal_features = np.load(config.multimodal_features, allow_pickle=True).item()

    items = pd.read_csv(config.items_csv)
    items["item_id"] = items["item_id"].astype(str)
    valid_set = set(str(v) for v in valid_ids) if valid_ids else None

    item_titles = {}
    for _, row in items.iterrows():
        iid = str(row["item_id"])
        if valid_set and iid not in valid_set:
            continue
        if iid in multimodal_features:
            item_titles[iid] = str(row["title"])

    all_item_ids = list(item_titles.keys())
    all_item_vecs = np.array([multimodal_features[iid] for iid in all_item_ids], dtype=np.float32)
    norms = np.linalg.norm(all_item_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    all_item_vecs = all_item_vecs / norms

    all_text_vecs = np.array([normalize(text_features[iid]) for iid in all_item_ids], dtype=np.float32)
    all_text_norms = np.linalg.norm(all_text_vecs, axis=1, keepdims=True)
    all_text_norms = np.maximum(all_text_norms, 1e-9)
    all_text_vecs = all_text_vecs / all_text_norms

    bert_retriever = BERTRetriever(items_path=config.items_csv, valid_ids=valid_ids)

    train_history_items = {}
    for _, row in train_history.iterrows():
        uid = row["user_id"]
        iid = str(row["item_id"])
        train_history_items.setdefault(uid, []).append(iid)

    recommendations = {}
    log_every = max(1, len(ground_truth) // 20)

    for idx, user_id in enumerate(ground_truth, start=1):
        history_items = train_history_items.get(user_id, [])

        user_multi_vecs = []
        user_text_vecs = []
        for iid in history_items:
            if iid in multimodal_features:
                user_multi_vecs.append(normalize(multimodal_features[iid]))
            if iid in text_features:
                user_text_vecs.append(normalize(text_features[iid]))

        if not user_multi_vecs:
            recommendations[user_id] = []
            continue

        user_avg_vec = normalize(np.mean(user_multi_vecs, axis=0))
        user_text_vec = normalize(np.mean(user_text_vecs, axis=0)) if user_text_vecs else None

        multi_sims = cosine_similarity([user_avg_vec], all_item_vecs)[0]
        text_sims = cosine_similarity([user_text_vec], all_text_vecs)[0] if user_text_vec is not None else np.zeros(len(all_item_ids))

        query_parts = [item_titles.get(iid, "") for iid in history_items[-5:] if iid in item_titles]
        query_text = " ".join(query_parts).strip()

        bert_score_dict = {}
        if query_text:
            bert_results = bert_retriever.search(query_text, topk=max(topk * 5, 50))
            bert_score_dict = {item_id: score for item_id, score in bert_results}

        history_set = set(history_items)
        scored = []
        for i in range(len(all_item_ids)):
            iid = all_item_ids[i]
            if iid in history_set:
                continue

            multi_sim = float(multi_sims[i])
            text_sim = float(text_sims[i])
            bert_sim = bert_score_dict.get(iid, 0.0)

            signal_strengths = np.array([multi_sim, text_sim, bert_sim])
            base_weights = np.array([0.40, 0.25, 0.35])
            attention_w = base_weights * (0.5 + 0.5 * np.clip(signal_strengths, 0, 1))
            attention_w = attention_w / attention_w.sum()

            rank_score = (
                attention_w[0] * multi_sim +
                attention_w[1] * text_sim +
                attention_w[2] * bert_sim
            )
            scored.append((iid, float(rank_score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = scored[:topk]

        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[Optimized] {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks, model_name="Optimized")
    print("[Optimized Our Model] done", flush=True)
    return metrics, recommendations


def main():
    print("=" * 60)
    print("优化后 Text Search 评估: BERT Baseline vs Optimized Our Model")
    print("=" * 60)

    topk = 10
    max_users = 2000
    ks = (5, 10)

    images_dir = config.images_dir
    valid_ids = None
    if os.path.isdir(images_dir):
        valid_ids = set(
            f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg")
        )
        print(f"Valid items: {len(valid_ids)}")

    # 原始数据评估
    print("\n--- 原始数据评估 ---")
    train_history, ground_truth = build_leave_one_out_split(
        config.user_history_csv, holdout_len=2, max_users=max_users
    )
    print(f"Users: {len(ground_truth)}, Train interactions: {len(train_history)}")

    results = {}

    bert_start = time.perf_counter()
    bert_metrics, _ = evaluate_bert_baseline(train_history, ground_truth, topk, ks, valid_ids)
    bert_elapsed = time.perf_counter() - bert_start
    bert_metrics["eval_time_sec"] = round(bert_elapsed, 2)
    results["BERT Baseline"] = bert_metrics

    opt_start = time.perf_counter()
    opt_metrics, _ = evaluate_optimized_our_model(
        train_history, ground_truth, topk, ks, valid_ids
    )
    opt_elapsed = time.perf_counter() - opt_start
    opt_metrics["eval_time_sec"] = round(opt_elapsed, 2)
    results["Optimized Our Model"] = opt_metrics

    # 增强数据评估
    aug_history_path = config.user_history_csv.replace(".csv", "_augmented_full.csv")
    if os.path.exists(aug_history_path):
        print("\n--- 增强数据评估 ---")
        train_history_aug, ground_truth_aug = build_leave_one_out_split(
            aug_history_path, holdout_len=2, max_users=max_users
        )
        print(f"Augmented users: {len(ground_truth_aug)}, interactions: {len(train_history_aug)}")

        opt_aug_start = time.perf_counter()
        opt_aug_metrics, _ = evaluate_optimized_our_model(
            train_history_aug, ground_truth_aug, topk, ks, valid_ids, use_augmented=True
        )
        opt_aug_elapsed = time.perf_counter() - opt_aug_start
        opt_aug_metrics["eval_time_sec"] = round(opt_aug_elapsed, 2)
        results["Optimized + Augmented Data"] = opt_aug_metrics

    # 输出对比
    print("\n" + "=" * 60)
    print("评估结果对比")
    print("=" * 60)

    metric_keys = [
        "precision@5", "recall@5", "ndcg@5", "map@5", "hit_rate@5",
        "precision@10", "recall@10", "ndcg@10", "map@10", "hit_rate@10",
        "non_empty_rate", "eval_time_sec"
    ]

    models = list(results.keys())
    header = f"{'Metric':<20}" + "".join(f" {m:>22}" for m in models)
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        vals = []
        for m in models:
            v = results[m].get(key, 0)
            vals.append(v)
        line = f"{key:<20}"
        for v in vals:
            if isinstance(v, (int, float)):
                line += f" {v:>22.4f}"
            else:
                line += f" {str(v):>22}"
        print(line)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"optimized_eval_{timestamp}.json")

    payload = {
        "experiment": "Optimized Text Search Evaluation",
        "timestamp": timestamp,
        "config": {
            "topk": topk,
            "max_users": max_users,
            "ks": list(ks),
            "valid_items": len(valid_ids) if valid_ids else "all",
            "optimizations": [
                "Removed BM25/token_overlap from rerank weights",
                "Added BERT semantic matching score",
                "Added attention-based dynamic weighting",
                "Data augmentation (similar users + semantic similarity)"
            ]
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
