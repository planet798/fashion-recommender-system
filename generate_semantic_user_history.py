import argparse
import random

import numpy as np
import pandas as pd

from bm25_baseline import BM25Retriever


def _normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_semantic_history(
    items_path,
    feature_path="data/multimodal_features.npy",
    num_users=80,
    history_len=8,
    top_pool=80,
    mixed_ratio=0.25,
    seed=42
):
    rng = random.Random(seed)
    retriever = BM25Retriever(items_path=items_path)
    features = np.load(feature_path, allow_pickle=True).item()
    item_ids = list(retriever.item_titles.keys())
    records = []

    for user_id in range(1, num_users + 1):
        anchor_item = rng.choice(item_ids)
        anchor_title = retriever.item_titles[anchor_item]
        anchor_vec = _normalize(features[anchor_item])

        bm25_similar_items = [
            item_id
            for item_id, _ in retriever.search(anchor_title, topk=top_pool)
            if item_id != anchor_item
        ]
        multimodal_scores = []
        for item_id in item_ids:
            if item_id == anchor_item or item_id not in features:
                continue
            score = float(np.dot(anchor_vec, _normalize(features[item_id])))
            multimodal_scores.append((item_id, score))
        multimodal_scores.sort(key=lambda x: x[1], reverse=True)
        multimodal_similar_items = [item_id for item_id, _ in multimodal_scores[:top_pool]]

        history = [anchor_item]
        candidate_pool = []
        seen_candidates = set()

        for item_id in bm25_similar_items + multimodal_similar_items:
            if item_id in seen_candidates:
                continue
            candidate_pool.append(item_id)
            seen_candidates.add(item_id)

        if rng.random() < mixed_ratio:
            second_anchor = rng.choice(item_ids)
            second_title = retriever.item_titles[second_anchor]
            second_vec = _normalize(features[second_anchor])
            second_similar = [
                item_id
                for item_id, _ in retriever.search(second_title, topk=max(20, top_pool // 2))
                if item_id not in history and item_id != second_anchor
            ]
            second_multimodal = []
            for item_id in item_ids:
                if item_id in history or item_id == second_anchor or item_id not in features:
                    continue
                score = float(np.dot(second_vec, _normalize(features[item_id])))
                second_multimodal.append((item_id, score))
            second_multimodal.sort(key=lambda x: x[1], reverse=True)
            candidate_pool = (
                candidate_pool[: max(1, history_len - 3)]
                + [second_anchor]
                + second_similar[: max(1, history_len - 3)]
                + [item_id for item_id, _ in second_multimodal[: max(1, history_len - 3)]]
            )

        deduped_pool = []
        seen = set(history)
        for item_id in candidate_pool:
            if item_id in seen:
                continue
            deduped_pool.append(item_id)
            seen.add(item_id)

        needed = max(history_len - len(history), 0)
        if len(deduped_pool) < needed:
            fallback = [item_id for item_id in item_ids if item_id not in seen]
            rng.shuffle(fallback)
            deduped_pool.extend(fallback[: needed - len(deduped_pool)])

        history.extend(deduped_pool[:needed])

        for timestamp, item_id in enumerate(history, start=1):
            records.append({
                "user_id": user_id,
                "item_id": int(item_id),
                "timestamp": timestamp
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate semantically coherent synthetic user histories.")
    parser.add_argument("--items-path", default="data/items.csv")
    parser.add_argument("--feature-path", default="data/multimodal_features.npy")
    parser.add_argument("--output-path", default="data/user_history.csv")
    parser.add_argument("--num-users", type=int, default=80)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--top-pool", type=int, default=80)
    parser.add_argument("--mixed-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_semantic_history(
        items_path=args.items_path,
        feature_path=args.feature_path,
        num_users=args.num_users,
        history_len=args.history_len,
        top_pool=args.top_pool,
        mixed_ratio=args.mixed_ratio,
        seed=args.seed
    )
    df.to_csv(args.output_path, index=False)
    print(f"user_history.csv generated at: {args.output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
