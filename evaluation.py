import math

import numpy as np


def load_ground_truth(path=None):
    from data_config import config
    path = path or os.path.join(config.root, "ground_truth.npy")
    return np.load(path, allow_pickle=True).item()


def _recommended_items(recommended, k):
    return [str(item) for item, _ in recommended[:k]]


def precision_at_k(recommended, ground_truth_items, k=10):
    gt_items = {str(item) for item in ground_truth_items}
    rec_items = _recommended_items(recommended, k)
    if k == 0:
        return 0.0
    hit = len(set(rec_items) & gt_items)
    return hit / k


def recall_at_k(recommended, ground_truth_items, k=10):
    gt_items = {str(item) for item in ground_truth_items}
    if not gt_items:
        return 0.0
    rec_items = _recommended_items(recommended, k)
    hit = len(set(rec_items) & gt_items)
    return hit / len(gt_items)


def ndcg_at_k(recommended, ground_truth_items, k=10):
    gt_items = {str(item) for item in ground_truth_items}
    dcg = 0.0

    for rank, item_id in enumerate(_recommended_items(recommended, k), start=1):
        if item_id in gt_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gt_items), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def average_precision_at_k(recommended, ground_truth_items, k=10):
    gt_items = {str(item) for item in ground_truth_items}
    if not gt_items:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, item_id in enumerate(_recommended_items(recommended, k), start=1):
        if item_id in gt_items:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0

    return precision_sum / min(len(gt_items), k)


def hit_rate_at_k(recommended, ground_truth_items, k=10):
    gt_items = {str(item) for item in ground_truth_items}
    return 1.0 if len(set(_recommended_items(recommended, k)) & gt_items) > 0 else 0.0


def evaluate_recommendations(recommendations_by_user, ground_truth, ks=(5, 10), model_name=""):
    metrics = {}

    for k in ks:
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
        metrics[f"map@{k}"] = 0.0
        metrics[f"hit_rate@{k}"] = 0.0

    user_count = 0
    non_empty_count = 0

    # --- 诊断日志 ---
    MAX_LOG_USERS = 5
    logged_users = 0

    for user_id, ground_truth_items in ground_truth.items():
        recommended = recommendations_by_user.get(user_id, [])
        user_count += 1
        if recommended:
            non_empty_count += 1

        # --- 诊断日志 ---
        if logged_users < MAX_LOG_USERS:
            gt_set = set(map(str, ground_truth_items))
            rec_set_k5 = set(_recommended_items(recommended, 5))
            hit_items = gt_set.intersection(rec_set_k5)
            
            print("-" * 50)
            print(f"DEBUG LOG: Model='{model_name}', User='{user_id}'")
            print(f"  Ground Truth: {gt_set}")
            print(f"  Recommended @5: {rec_set_k5}")
            if hit_items:
                print(f"  !!! HIT @5: {hit_items} !!!")
            else:
                print(f"  --- MISS @5 ---")
            
            # 打印更长的召回列表以供分析
            if model_name in ["FAISS", "Hybrid", "Learned Ranker"]:
                 print(f"  Full Recall (top 20): {_recommended_items(recommended, 20)}")

            logged_users += 1
            if logged_users == MAX_LOG_USERS:
                print("-" * 50)
                print("DEBUG LOG: Max log users reached. No more logs will be printed.")
                print("-" * 50)


        for k in ks:
            metrics[f"precision@{k}"] += precision_at_k(recommended, ground_truth_items, k=k)
            metrics[f"recall@{k}"] += recall_at_k(recommended, ground_truth_items, k=k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(recommended, ground_truth_items, k=k)
            metrics[f"map@{k}"] += average_precision_at_k(recommended, ground_truth_items, k=k)
            metrics[f"hit_rate@{k}"] += hit_rate_at_k(recommended, ground_truth_items, k=k)

    if user_count == 0:
        return metrics

    for key in list(metrics.keys()):
        metrics[key] /= user_count
    
    metrics["non_empty_rate"] = non_empty_count / user_count if user_count > 0 else 0.0
    metrics["user_count"] = user_count
    return metrics