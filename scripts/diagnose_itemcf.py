from pathlib import Path

import pandas as pd

from fashionrec.models.itemcf import ItemCFRecommender


ROOT = Path("data/processed/dev_split")


def main():
    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    validation = pd.read_parquet(
        ROOT / "validation.parquet"
    )

    model = ItemCFRecommender(
        top_neighbors=100,
    )

    model.fit(train)

    histories = (
        train.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    train_items = set(
        train["item_id"].astype(str)
    )

    # ----------------------------------
    # Item neighbor statistics
    # ----------------------------------

    neighbor_counts = pd.Series(
        {
            item: len(neighbors)
            for item, neighbors
            in model.neighbors.items()
        }
    )

    print()
    print("=" * 60)
    print("ItemCF Diagnostics")
    print("=" * 60)

    print()
    print("Items in train:")
    print(f"{len(train_items):,}")

    print()
    print("Items with at least one neighbor:")
    print(f"{len(model.neighbors):,}")

    coverage = (
        len(model.neighbors)
        / len(train_items)
    )

    print(
        f"Neighbor coverage: "
        f"{coverage:.4%}"
    )

    if not neighbor_counts.empty:
        print()
        print("Neighbor count distribution:")
        print(
            neighbor_counts.describe()
        )

    # ----------------------------------
    # User recommendation availability
    # ----------------------------------

    empty_users = 0
    total_candidates = 0

    warm_users = 0
    reachable_warm = 0

    for row in validation.itertuples(
        index=False
    ):
        user_id = row.user_id
        target = str(row.item_id)

        history = histories[user_id]

        recommendations = model.recommend(
            history,
            k=100,
        )

        total_candidates += len(
            recommendations
        )

        if not recommendations:
            empty_users += 1

        if target in train_items:
            warm_users += 1

            reachable = False

            for history_item in history:

                neighbors = model.neighbors.get(
                    str(history_item),
                    [],
                )

                neighbor_items = {
                    item
                    for item, _
                    in neighbors
                }

                if target in neighbor_items:
                    reachable = True
                    break

            if reachable:
                reachable_warm += 1

    num_users = len(validation)

    print()
    print("User recommendation diagnostics")
    print("-" * 50)

    print(
        f"Users with empty recommendations: "
        f"{empty_users:,}"
    )

    print(
        f"Empty recommendation rate: "
        f"{empty_users / num_users:.4%}"
    )

    print(
        f"Average generated candidates: "
        f"{total_candidates / num_users:.2f}"
    )

    print()
    print("Warm target reachability")
    print("-" * 50)

    print(
        f"Warm users: "
        f"{warm_users:,}"
    )

    print(
        f"Reachable warm targets: "
        f"{reachable_warm:,}"
    )

    print(
        f"Warm reachability rate: "
        f"{reachable_warm / warm_users:.4%}"
    )


if __name__ == "__main__":
    main()
