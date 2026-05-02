# -*- coding: utf-8 -*-
"""
BERT Baseline vs Our Model 对比评估脚本 (优化版)
BERT: all-MiniLM-L6-v2 语义检索基线
Our Model (优化版): Two-Tower召回(FAISS HNSW) + DIN排序 + LambdaMART

优化内容:
1. 召回层: 使用FaissRecallV2的Two-Tower结构 + FAISS HNSW索引
2. 排序层: DIN注意力机制 + LambdaMART (如可用)
3. 评估指标: Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K
"""

import os
import json
import time
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
os.environ["HF_HOME"] = "D:\\HF_HOME"
os.environ["TRANSFORMERS_CACHE"] = "D:\\HF_HOME\\transformers"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd
from datetime import datetime

from bert_baseline import BERTRetriever
from evaluation import evaluate_recommendations
from data_config import config
from recall_faiss_v2 import FaissRecallV2
from user_model import UserInterestModel


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


def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


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


def evaluate_our_model_optimized(train_history, ground_truth, topk, ks, valid_ids):
    """
    Our Model (优化版): Two-Tower召回 + DIN排序
    1. 召回层: FaissRecallV2 (多兴趣文本塔 + 图像塔, RRF融合)
    2. 排序层: DIN注意力加权 (如可用) + 多特征重排
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"[Our Model Optimized] start: users={len(ground_truth)} topk={topk}", flush=True)

    print("[Our Model] Loading features and building indices...", flush=True)

    multimodal_features = np.load(config.multimodal_features, allow_pickle=True).item()

    print("[Our Model] Initializing UserInterestModel...", flush=True)
    user_model = UserInterestModel(
        history_path=config.user_history_csv,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    )

    print("[Our Model] Initializing FaissRecallV2 (Two-Tower + FAISS HNSW)...", flush=True)
    faiss_recall = FaissRecallV2(
        user_model=user_model,
        text_index_path=config.text_index,
        text_ids_path=config.text_ids,
        image_index_path=config.image_index,
        image_ids_path=config.image_ids
    )

    din_ranker = None
    try:
        from din_ranking_model import DINRanker
        print("[Our Model] Initializing DIN Ranker...", flush=True)
        din_ranker = DINRanker(
            text_feature_path=config.text_features,
            image_feature_path=config.image_features,
            multimodal_feature_path=config.multimodal_features,
            history_path=config.user_history_csv
        )
        print("[Our Model] DIN Ranker loaded successfully", flush=True)
    except Exception as e:
        print(f"[Our Model] DIN Ranker not available: {e}", flush=True)

    train_history_items = {}
    for _, row in train_history.iterrows():
        uid = row["user_id"]
        iid = str(row["item_id"])
        train_history_items.setdefault(uid, []).append(iid)

    recommendations = {}
    log_every = max(1, len(ground_truth) // 20)
    recall_topk = max(100, topk * 10)

    for idx, user_id in enumerate(ground_truth, start=1):
        history_items = train_history_items.get(user_id, [])

        candidates = faiss_recall.recall_by_user(user_id, topk=recall_topk)

        if not candidates:
            recommendations[user_id] = []
            if idx % log_every == 0 or idx == len(ground_truth):
                print(f"[Our Model] {idx}/{len(ground_truth)} users", flush=True)
            continue

        candidate_ids = [cid for cid, _ in candidates]

        if din_ranker is not None:
            try:
                reranked = din_ranker.rerank_user_candidates(user_id, candidate_ids, topk=topk)
                if reranked and isinstance(reranked[0], dict):
                    recommendations[user_id] = [(r["item_id"], r["final_score"]) for r in reranked]
                else:
                    recommendations[user_id] = reranked[:topk]
            except Exception as e:
                print(f"[Our Model] DIN rerank failed for user {user_id}: {e}")
                history_set = set(history_items)
                scored = [(cid, float(score)) for cid, score in candidates if cid not in history_set]
                recommendations[user_id] = scored[:topk]
        else:
            history_set = set(history_items)
            scored = []
            for cid, score in candidates:
                if cid not in history_set:
                    scored.append((cid, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            recommendations[user_id] = scored[:topk]

        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[Our Model] {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks, model_name="OurModel(DIN+FAISS)")
    print("[Our Model Optimized] done", flush=True)
    return metrics, recommendations


def main():
    print("=" * 60)
    print("BERT Baseline vs Our Model (Optimized) 对比评估")
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
        print(f"Valid items (with images): {len(valid_ids)}")

    print(f"Building leave-one-out split (max_users={max_users})...", flush=True)
    train_history, ground_truth = build_leave_one_out_split(
        config.user_history_csv, holdout_len=2, max_users=max_users
    )
    print(f"Users: {len(ground_truth)}, Train interactions: {len(train_history)}")

    results = {}

    bert_start = time.perf_counter()
    bert_metrics, bert_recs = evaluate_bert_baseline(
        train_history, ground_truth, topk, ks, valid_ids
    )
    bert_elapsed = time.perf_counter() - bert_start
    bert_metrics["eval_time_sec"] = round(bert_elapsed, 2)
    results["BERT Baseline"] = bert_metrics

    our_start = time.perf_counter()
    our_metrics, our_recs = evaluate_our_model_optimized(
        train_history, ground_truth, topk, ks, valid_ids
    )
    our_elapsed = time.perf_counter() - our_start
    our_metrics["eval_time_sec"] = round(our_elapsed, 2)
    results["Our Model (DIN+FAISS)"] = our_metrics

    print("\n" + "=" * 60)
    print("评估结果对比")
    print("=" * 60)

    metric_keys = [
        "precision@5", "recall@5", "ndcg@5", "map@5", "hit_rate@5",
        "precision@10", "recall@10", "ndcg@10", "map@10", "hit_rate@10",
        "non_empty_rate", "eval_time_sec"
    ]

    header = f"{'Metric':<20} {'BERT Baseline':>18} {'Our Model':>18} {'Delta':>12}"
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        bert_val = bert_metrics.get(key, 0)
        our_val = our_metrics.get(key, 0)
        if isinstance(bert_val, (int, float)) and isinstance(our_val, (int, float)):
            delta = our_val - bert_val
            sign = "+" if delta > 0 else ""
            print(f"{key:<20} {bert_val:>18.4f} {our_val:>18.4f} {sign}{delta:>11.4f}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"bert_vs_our_optimized_{timestamp}.json")

    payload = {
        "experiment": "BERT Baseline vs Our Model (Optimized)",
        "timestamp": timestamp,
        "config": {
            "topk": topk,
            "max_users": max_users,
            "ks": list(ks),
            "valid_items": len(valid_ids) if valid_ids else "all",
            "bert_model": "all-MiniLM-L6-v2",
            "recall": "FaissRecallV2 (Two-Tower + FAISS HNSW)",
            "ranking": "DIN Ranker (if available)",
            "our_model": "SentenceTransformer + CLIP + Multimodal Fusion + DIN"
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
