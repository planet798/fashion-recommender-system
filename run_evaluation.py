import argparse
import json
import os
import tempfile
import time
from datetime import datetime

import pandas as pd

from bm25_baseline import BM25Retriever
from bert_baseline import BERTRetriever
from evaluation import evaluate_recommendations


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


def select_metric_ks(topk):
    ks = {topk}
    if topk >= 5:
        ks.add(5)
    if topk >= 10:
        ks.add(10)
    return tuple(sorted(ks))


def add_diagnostics(metrics, recommendations, ground_truth, topk):
    non_empty = sum(1 for rows in recommendations.values() if len(rows) > 0)
    hit_users = 0

    for user_id, gt_items in ground_truth.items():
        rec_items = {item_id for item_id, _ in recommendations.get(user_id, [])[:topk]}
        if rec_items & set(gt_items):
            hit_users += 1

    metrics["non_empty_rate"] = non_empty / max(len(ground_truth), 1)
    metrics["hit_users"] = hit_users
    return metrics


def rank_of_first_hit(rows, target_items):
    for rank, row in enumerate(rows, start=1):
        item_id = row["item_id"] if isinstance(row, dict) else row[0]
        if str(item_id) in target_items:
            return rank
    return None


def add_candidate_diagnostics(metrics, candidate_rows_by_user, ground_truth, reranked_rows_by_user=None):
    candidate_hits = 0
    candidate_ranks = []
    candidate_sizes = []

    rerank_hits = 0
    rerank_ranks = []

    for user_id, gt_items in ground_truth.items():
        gt_set = {str(item_id) for item_id in gt_items}
        candidates = candidate_rows_by_user.get(user_id, [])
        candidate_sizes.append(len(candidates))
        candidate_rank = rank_of_first_hit(candidates, gt_set)
        if candidate_rank is not None:
            candidate_hits += 1
            candidate_ranks.append(candidate_rank)

        if reranked_rows_by_user is not None:
            reranked = reranked_rows_by_user.get(user_id, [])
            rerank_rank = rank_of_first_hit(reranked, gt_set)
            if rerank_rank is not None:
                rerank_hits += 1
                rerank_ranks.append(rerank_rank)

    user_count = max(len(ground_truth), 1)
    metrics["positive_in_candidates_rate"] = candidate_hits / user_count
    metrics["candidate_size_avg"] = sum(candidate_sizes) / max(len(candidate_sizes), 1)
    metrics["positive_rank_in_candidates_avg"] = (
        sum(candidate_ranks) / len(candidate_ranks) if candidate_ranks else None
    )

    if reranked_rows_by_user is not None:
        metrics["positive_in_reranked_rate"] = rerank_hits / user_count
        metrics["positive_rank_after_rerank_avg"] = (
            sum(rerank_ranks) / len(rerank_ranks) if rerank_ranks else None
        )

    return metrics


def progress_interval(total_users):
    if total_users <= 20:
        return 1
    if total_users <= 100:
        return 5
    return 20


def parse_models(raw_models):
    alias_map = {
        "bm25": "bm25",
        "bert": "bert",
        "faiss": "faiss",
        "hybrid": "hybrid",
        "learned": "learned",
        "learned_ranker": "learned",
    }
    selected = []
    for token in str(raw_models).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in alias_map:
            raise ValueError(f"unknown model name: {token}")
        normalized = alias_map[token]
        if normalized not in selected:
            selected.append(normalized)
    return selected


def build_eval_item_subset(items_path, train_history, ground_truth):
    eval_item_ids = set(train_history["item_id"].astype(str).tolist())
    for gt_items in ground_truth.values():
        eval_item_ids.update(str(item_id) for item_id in gt_items)

    items = pd.read_csv(items_path)
    items["item_id"] = items["item_id"].astype(str)
    subset = items[items["item_id"].isin(eval_item_ids)].copy()
    return subset


def evaluate_bm25(train_history, ground_truth, items_path, topk, ks, valid_ids=None):
    print(f"[BM25] start: users={len(ground_truth)} topk={topk}", flush=True)
    retriever = BM25Retriever(items_path=items_path, valid_ids=valid_ids)
    recommendations = {}
    log_every = progress_interval(len(ground_truth))

    for idx, user_id in enumerate(ground_truth, start=1):
        recommendations[user_id] = retriever.recommend_for_user(train_history, user_id, topk=topk)
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[BM25] processed {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    print("[BM25] done", flush=True)
    return add_diagnostics(metrics, recommendations, ground_truth, topk)


def evaluate_bert(train_history, ground_truth, items_path, topk, ks, valid_ids=None, **kwargs):
    from data_config import config

    print(f"[BERT] start: users={len(ground_truth)} topk={topk}", flush=True)
    retriever = BERTRetriever(items_path=items_path, valid_ids=valid_ids)
    recommendations = {}
    log_every = progress_interval(len(ground_truth))

    for idx, user_id in enumerate(ground_truth, start=1):
        recommendations[user_id] = retriever.recommend_for_user(train_history, user_id, topk=topk)
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[BERT] processed {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    print("[BERT] done", flush=True)
    return add_diagnostics(metrics, recommendations, ground_truth, topk)


def evaluate_faiss(train_history_path, ground_truth, topk, ks, **kwargs):
    from recall_faiss_v2 import FaissRecallV2
    from user_model import UserInterestModel
    from data_config import config

    print(f"[FAISS] start: users={len(ground_truth)} topk={topk}", flush=True)
    
    user_model = UserInterestModel(
        history_path=train_history_path,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    )
    
    recall = FaissRecallV2(
        user_model=user_model,
        text_index_path=config.text_index,
        text_ids_path=config.text_ids,
        image_index_path=config.image_index,
        image_ids_path=config.image_ids
    )
    
    recommendations = {}
    log_every = progress_interval(len(ground_truth))

    for idx, user_id in enumerate(ground_truth, start=1):
        history_items = set(recall.user_model.get_user_history(user_id, max_len=20))
        results = recall.recall_by_user(user_id=user_id, topk=max(20, topk * 3))
        recommendations[user_id] = [
            (item_id, score)
            for item_id, score in results
            if item_id not in history_items
        ][:topk]
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[FAISS] processed {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    print("[FAISS] done", flush=True)
    return add_diagnostics(metrics, recommendations, ground_truth, topk)


def evaluate_hybrid(train_history_path, ground_truth, items_path, topk, ks, **kwargs):
    from hybrid_recall_v3 import HybridRecallV3
    from recall_faiss_v2 import FaissRecallV2
    from user_model import UserInterestModel
    from bm25_baseline import BM25Retriever
    from data_config import config

    print(f"[Hybrid] start: users={len(ground_truth)} topk={topk}", flush=True)
    
    user_model = UserInterestModel(
        history_path=train_history_path,
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
    
    bm25_retriever = BM25Retriever(items_path=items_path)

    recall = HybridRecallV3(
        faiss_recall=faiss_recall,
        bm25=bm25_retriever,
        mode="hybrid"
    )
    
    recommendations = {}
    candidate_rows_by_user = {}
    log_every = progress_interval(len(ground_truth))

    for idx, user_id in enumerate(ground_truth, start=1):
        candidate_rows = recall.recall_by_user(user_id=user_id, topk=max(20, topk * 3), return_details=True)
        candidate_rows_by_user[user_id] = candidate_rows
        recommendations[user_id] = [(row["item_id"], row["final_score"]) for row in candidate_rows[:topk]]
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[Hybrid] processed {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    print("[Hybrid] done", flush=True)
    metrics = add_diagnostics(metrics, recommendations, ground_truth, topk)
    return add_candidate_diagnostics(metrics, candidate_rows_by_user, ground_truth)


def evaluate_learned_ranker(train_history_path, ground_truth, items_path, topk, model_path, ks, **kwargs):
    from hybrid_recall_v3 import HybridRecallV3
    from recall_faiss_v2 import FaissRecallV2
    from user_model import UserInterestModel
    from bm25_baseline import BM25Retriever
    from ranking_model import PairwiseFeatureRanker
    from data_config import config

    print(f"[LearnedRanker] start: users={len(ground_truth)} topk={topk}", flush=True)

    user_model = UserInterestModel(
        history_path=train_history_path,
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
    
    bm25_retriever = BM25Retriever(items_path=items_path)

    recall = HybridRecallV3(
        faiss_recall=faiss_recall,
        bm25=bm25_retriever,
        mode="hybrid"
    )

    ranker = PairwiseFeatureRanker(
        model_path=model_path,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    ).load()
    recommendations = {}
    candidate_rows_by_user = {}
    reranked_rows_by_user = {}
    log_every = progress_interval(len(ground_truth))

    for idx, user_id in enumerate(ground_truth, start=1):
        candidates = recall.recall_by_user(user_id=user_id, topk=max(30, topk * 3), return_details=True)
        
        # --- 注入 Ground Truth ---
        user_ground_truth = ground_truth.get(user_id, [])
        gt_in_candidates = {str(c["item_id"]) for c in candidates}
        
        for gt_item_id in user_ground_truth:
            if str(gt_item_id) not in gt_in_candidates:
                # For ground truth items not found by recall, add them to the list
                # The ranker will need to fetch their features manually
                candidates.append({"item_id": str(gt_item_id)})
        # --- 结束注入 ---

        candidate_rows_by_user[user_id] = candidates

        reranked_rows = ranker.rerank_candidates(
            user_id=user_id,
            candidates=candidates,
            user_model=recall.faiss.user_model,
            bm25=recall.bm25,
            topk=topk,
            ground_truth=user_ground_truth
        )
        reranked_rows_by_user[user_id] = reranked_rows
        recommendations[user_id] = [(row["item_id"], row["final_score"]) for row in reranked_rows]
        if idx % log_every == 0 or idx == len(ground_truth):
            print(f"[LearnedRanker] processed {idx}/{len(ground_truth)} users", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks, model_name=kwargs.get("model_name", ""))
    print("[LearnedRanker] done", flush=True)
    metrics = add_diagnostics(metrics, recommendations, ground_truth, topk)
    return add_candidate_diagnostics(metrics, candidate_rows_by_user, ground_truth, reranked_rows_by_user)


def main():
    from data_config import config
    parser = argparse.ArgumentParser(description="Evaluate BM25, FAISS and Hybrid recommendation baselines.")
    parser.add_argument("--items-path", default=config.items_csv)
    parser.add_argument("--history-path", default=config.user_history_csv)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--ranking-model-path", default=config.ranking_model)
    parser.add_argument("--holdout-len", type=int, default=2)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--feature-path", default=config.multimodal_features)
    parser.add_argument("--index-path", default=config.text_index)
    parser.add_argument("--ids-path", default=config.text_ids)
    parser.add_argument("--text-model-path", default="models/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--models", default="bm25,bert,faiss,hybrid,learned")
    parser.add_argument("--filter-items-to-history", action="store_true")
    parser.add_argument("--images-dir", default="", help="Images dir to filter valid item IDs")
    args = parser.parse_args()

    selected_models = parse_models(args.models)
    print(f"[Config] models={selected_models}", flush=True)

    valid_ids = None
    if args.images_dir and os.path.isdir(args.images_dir):
        valid_ids = set(
            f.replace(".jpg", "") for f in os.listdir(args.images_dir) if f.endswith(".jpg")
        )
        print(f"[Config] valid_ids from images: {len(valid_ids)}", flush=True)

    split_start = time.perf_counter()
    train_history, ground_truth = build_leave_one_out_split(
        args.history_path,
        holdout_len=args.holdout_len,
        max_users=args.max_users if args.max_users > 0 else None
    )
    print(
        f"[Split] users={len(ground_truth)} interactions={len(train_history)} holdout_len={args.holdout_len} "
        f"elapsed={time.perf_counter() - split_start:.2f}s",
        flush=True
    )
    ks = select_metric_ks(args.topk)
    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as temp_file:
        train_history.to_csv(temp_file.name, index=False)
        train_history_path = temp_file.name

    eval_items_path = args.items_path
    eval_items_temp_path = None

    if args.filter_items_to_history:
        subset_start = time.perf_counter()
        eval_items = build_eval_item_subset(args.items_path, train_history, ground_truth)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as temp_items_file:
            eval_items.to_csv(temp_items_file.name, index=False)
            eval_items_temp_path = temp_items_file.name
            eval_items_path = temp_items_file.name
        print(
            f"[ItemsSubset] original_path={args.items_path} subset_items={len(eval_items)} "
            f"elapsed={time.perf_counter() - subset_start:.2f}s",
            flush=True
        )

    def safe_eval(name, fn, *eval_args):
        started = time.perf_counter()
        print(f"[{name}] loading and evaluating...", flush=True)
        try:
            result = fn(*eval_args)
            print(f"[{name}] finished in {time.perf_counter() - started:.2f}s", flush=True)
            return result
        except ModuleNotFoundError as exc:
            print(f"[{name}] skipped after {time.perf_counter() - started:.2f}s", flush=True)
            return {
                "status": "skipped",
                "reason": f"missing dependency: {exc.name}"
            }
        except ValueError as exc:
            if "ranking model feature mismatch" in str(exc):
                print(f"[{name}] skipped after {time.perf_counter() - started:.2f}s", flush=True)
                return {
                    "status": "skipped",
                    "reason": str(exc)
                }
            raise
        except Exception:
            print(f"[{name}] failed after {time.perf_counter() - started:.2f}s", flush=True)
            raise

    try:
        results = {}

        if "bm25" in selected_models:
            results["bm25"] = safe_eval(
                "bm25",
                evaluate_bm25,
                train_history,
                ground_truth,
                eval_items_path,
                args.topk,
                ks,
                valid_ids
            )

        if "bert" in selected_models:
            results["bert"] = safe_eval(
                "bert",
                evaluate_bert,
                train_history,
                ground_truth,
                eval_items_path,
                args.topk,
                ks,
                valid_ids
            )

        if "faiss" in selected_models:
            results["faiss_multi_interest"] = safe_eval(
                "faiss_multi_interest",
                evaluate_faiss,
                train_history_path,
                ground_truth,
                args.topk,
                ks
            )

        if "hybrid" in selected_models:
            results["hybrid_rrf"] = safe_eval(
                "hybrid_rrf",
                evaluate_hybrid,
                train_history_path,
                ground_truth,
                eval_items_path,
                args.topk,
                ks
            )

        if "learned" in selected_models and os.path.exists(args.ranking_model_path):
            results["hybrid_learned_ranker"] = safe_eval(
                "hybrid_learned_ranker",
                evaluate_learned_ranker,
                train_history_path,
                ground_truth,
                eval_items_path,
                args.topk,
                args.ranking_model_path,
                ks
            )
    finally:
        if os.path.exists(train_history_path):
            os.remove(train_history_path)
        if eval_items_temp_path and os.path.exists(eval_items_temp_path):
            os.remove(eval_items_temp_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"evaluation_{timestamp}.json")
    payload = {
        "items_path": args.items_path,
        "history_path": args.history_path,
        "eval_items_path": eval_items_path,
        "topk": args.topk,
        "holdout_len": args.holdout_len,
        "max_users": args.max_users,
        "filter_items_to_history": args.filter_items_to_history,
        "split": "leave-last-n-out",
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
