from collections import defaultdict

import numpy as np
import pandas as pd


TRAIN_PATH = "data/processed/collab_5core/train.parquet"
SAMPLE_PATH = "data/processed/eval/validation_warm_10k_seed42.parquet"

K = 10


def main():
    print("Loading train...")
    train = pd.read_parquet(
        TRAIN_PATH,
        columns=["user_id", "item_id"],
    )

    print("Loading fixed validation sample...")
    sample = pd.read_parquet(
        SAMPLE_PATH,
    )

    # Global item popularity.
    popularity = (
        train.groupby("item_id")
        .size()
        .reset_index(name="count")
        .sort_values(
            ["count", "item_id"],
            ascending=[False, True],
        )
    )

    ranked_items = popularity[
        "item_id"
    ].tolist()

    sample_users = set(
        sample["user_id"]
    )

    # Only histories needed by these 10k users.
    selected_train = train[
        train["user_id"].isin(sample_users)
    ]

    seen_items = defaultdict(set)

    for user, item in zip(
        selected_train["user_id"],
        selected_train["item_id"],
    ):
        seen_items[user].add(item)

    hits = 0
    ndcg_sum = 0.0
    mrr_sum = 0.0

    print("Evaluating popularity...")

    for user, target in zip(
        sample["user_id"],
        sample["item_id"],
    ):
        seen = seen_items[user]

        recommendations = []

        for item in ranked_items:
            if item in seen:
                continue

            recommendations.append(item)

            if len(recommendations) == K:
                break

        try:
            rank = recommendations.index(
                target
            ) + 1
        except ValueError:
            continue

        hits += 1
        ndcg_sum += (
            1.0 / np.log2(rank + 1)
        )
        mrr_sum += 1.0 / rank

    n = len(sample)

    print()
    print("=" * 60)
    print("Popularity - Fixed Warm Validation Sample")
    print("=" * 60)
    print(f"Users   : {n:,}")
    print(f"HR@10   : {hits / n:.6f}")
    print(f"NDCG@10 : {ndcg_sum / n:.6f}")
    print(f"MRR@10  : {mrr_sum / n:.6f}")


if __name__ == "__main__":
    main()
