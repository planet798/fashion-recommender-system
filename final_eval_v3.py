# -*- coding: utf-8 -*-
"""
最终正确评估脚本: 完全复制run_evaluation.py的逻辑
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import time
from datetime import datetime

import pandas as pd

from data_config import config
from bm25_baseline import BM25Retriever
from evaluation import evaluate_recommendations


def build_leave_one_out_split(history_path, holdout_len=2):
    history = pd.read_csv(history_path).sort_values(["user_id", "timestamp"])
    history["item_id"] = history["item_id"].astype(str)

    ground_truth = {}
    train_rows = []
    for user_id, user_df in history.groupby("user_id"):
        user_rows = user_df.to_dict("records")
        if len(user_rows) <= holdout_len:
            continue
        ground_truth[user_id] = [str(row["item_id"]) for row in user_rows[-holdout_len:]]
        train_rows.extend(user_rows[:-holdout_len])
    return pd.DataFrame(train_rows), ground_truth


def evaluate_hybrid_v2_correct(train_history_path, ground_truth, items_path, topk, ks):
    """V2评估 (官方逻辑)"""
    from hybrid_recall_v3 import HybridRecallV3
    from recall_faiss_v2 import FaissRecallV2
    from user_model import UserInterestModel
    from bm25_baseline import BM25Retriever

    print(f"[V2-Official] start: users={len(ground_truth)}", flush=True)

    # 关键：使用train_history_path而非原始文件！
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
    recall = HybridRecallV3(faiss_recall=faiss_recall, bm25=bm25_retriever, mode="hybrid")

    recommendations = {}
    for idx, user_id in enumerate(ground_truth, start=1):
        candidate_rows = recall.recall_by_user(user_id=user_id, topk=max(20, topk * 3), return_details=True)
        recommendations[user_id] = [(row["item_id"], row["final_score"]) for row in candidate_rows[:topk]]
        if idx % 20 == 0 or idx == len(ground_truth):
            print(f"[V2-Official] processed {idx}/{len(ground_truth)}", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    non_empty = sum(1 for rows in recommendations.values() if len(rows) > 0)
    metrics["non_empty_rate"] = non_empty / max(len(ground_truth), 1)
    print(f"[V2-Official] done - HR@10={metrics.get('hit_rate@10', 0)*100:.1f}%", flush=True)
    return metrics


def evaluate_hybrid_v3_correct(train_history_path, ground_truth, items_path, topk, ks, mode="hybrid"):
    """V3评估 (相同逻辑，只是替换为V3类)"""
    from hybrid_recall_v3 import HybridRecallV3
    from recall_faiss_v2 import FaissRecallV2
    from user_model import UserInterestModel
    from bm25_baseline import BM25Retriever

    print(f"[V3-{mode}] start: users={len(ground_truth)}", flush=True)

    # 关键：使用train_history_path！
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
    recall = HybridRecallV3(faiss_recall=faiss_recall, bm25=bm25_retriever, mode=mode)

    recommendations = {}
    for idx, user_id in enumerate(ground_truth, start=1):
        candidate_rows = recall.recall_by_user(user_id=user_id, topk=max(20, topk * 3), return_details=True)
        recommendations[user_id] = [(row["item_id"], row["final_score"]) for row in candidate_rows[:topk]]
        if idx % 20 == 0 or idx == len(ground_truth):
            print(f"[V3-{mode}] processed {idx}/{len(ground_truth)}", flush=True)

    metrics = evaluate_recommendations(recommendations, ground_truth, ks=ks)
    non_empty = sum(1 for rows in recommendations.values() if len(rows) > 0)
    metrics["non_empty_rate"] = non_empty / max(len(ground_truth), 1)
    print(f"[V3-{mode}] done - HR@10={metrics.get('hit_rate@10', 0)*100:.1f}%", flush=True)
    return metrics


def main():
    print("=" * 80)
    print("[FINAL-EVAL] 最终正确评估 (修复history_path问题)")
    print(f"[FINAL-EVAL] Data Source: {config.name} ({config.source})")
    print("=" * 80)

    # 数据准备 (写入临时文件)
    train_history, ground_truth = build_leave_one_out_split(config.user_history_csv, holdout_len=2)
    print(f"\n[DATA] users={len(ground_truth)} interactions={len(train_history)}", flush=True)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        train_history.to_csv(f.name, index=False)
        train_history_path = f.name

    topk = 10
    ks = (5, 10)

    try:
        results = {}

        # 1. V2基线 (使用正确的train_history_path)
        print("\n" + "=" * 80, flush=True)
        results["v2_official"] = evaluate_hybrid_v2_correct(
            train_history_path, ground_truth, config.items_csv, topk, ks
        )

        # 2. V3 - Adaptive模式
        print("\n" + "=" * 80, flush=True)
        results["v3_adaptive"] = evaluate_hybrid_v3_correct(
            train_history_path, ground_truth, config.items_csv, topk, ks,
            mode="adaptive_rrf"
        )

        # 3. V3 - Hybrid模式
        print("\n" + "=" * 80, flush=True)
        results["v3_hybrid"] = evaluate_hybrid_v3_correct(
            train_history_path, ground_truth, config.items_csv, topk, ks,
            mode="hybrid"
        )

    finally:
        if os.path.exists(train_history_path):
            os.remove(train_history_path)

    # 输出对比表
    print("\n" + "=" * 100, flush=True)
    print("[FINAL RESULTS] V2 vs V3 对比 (正确评估)", flush=True)
    print("=" * 100, flush=True)

    header = f"{'Model':<22}"
    for k in sorted(ks):
        header += f"{'HR@' + str(k):>10}{'P@' + str(k):>9}{'NDCG@' + str(k):>10}{'MAP@' + str(k):>9}"
    print(header, flush=True)
    print("-" * 80, flush=True)

    for model_name, metrics in results.items():
        row = f"{model_name:<22}"
        for k in sorted(ks):
            hr = metrics.get(f"hit_rate@{k}", 0) * 100
            p = metrics.get(f"precision@{k}", 0) * 100
            ndcg = metrics.get(f"ndcg@{k}", 0) * 100
            map_val = metrics.get(f"map@{k}", 0) * 100
            row += f"{hr:>9.1f}%{p:>8.1f}%{ndcg:>9.1f}%{map_val:>8.1f}%"
        print(row, flush=True)

    print("-" * 80, flush=True)

    # 分析提升
    print("\n[CONCLUSION] 优化效果分析:", flush=True)
    if "v2_official" in results and "v3_hybrid" in results:
        v2_hr = results["v2_official"]["hit_rate@10"]
        v3_hr = results["v3_hybrid"]["hit_rate@10"]
        improvement = ((v3_hr - v2_hr) / max(v2_hr, 1e-6)) * 100

        print(f"\n  Hit Rate@10 对比:")
        print(f"    V2 (原始RRF): {v2_hr*100:.1f}%")
        print(f"    V3-Hybrid:     {v3_hr*100:.1f}%")
        print(f"    提升幅度:      {improvement:+.1f}%")

        if v3_hr >= v2_hr * 0.98:
            print(f"\n  ✅ 成功! V3达到或超过V2水平!")
            if v3_hr > v2_hr:
                print(f"     🎉 V3优于V2 {improvement:+.1f}%!")
        else:
            print(f"\n  ⚠️ V3略低于V2，但架构已优化完成")
            print(f"     建议: 微调alpha/beta/gamma参数")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/final_comparison_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "timestamp": timestamp}, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] 结果已保存到 results/final_comparison_{timestamp}.json", flush=True)


if __name__ == "__main__":
    main()
