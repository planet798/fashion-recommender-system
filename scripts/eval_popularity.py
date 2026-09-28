from pathlib import Path

import pandas as pd

from fashionrec.evaluation.evaluator import (
    evaluate_rankings,
    print_evaluation_report,
)
from fashionrec.models.popularity import (
    PopularityRecommender,
)


ROOT = Path("data/processed/dev_split")

KS = (5, 10, 20)


def main():
    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    validation = pd.read_parquet(
        ROOT / "validation.parquet"
    )

    model = PopularityRecommender()
    model.fit(train)

    histories = (
        train.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    recommendations = {}

    for row in validation.itertuples(
        index=False
    ):
        user_id = row.user_id

        recommendations[user_id] = (
            model.recommend(
                seen_items=histories[user_id],
                k=max(KS),
            )
        )

    ground_truth = {
        row.user_id: str(row.item_id)
        for row in validation.itertuples(
            index=False
        )
    }

    train_items = set(
        train["item_id"].astype(str)
    )

    results = evaluate_rankings(
        recommendations,
        ground_truth,
        train_items,
        ks=KS,
    )

    print_evaluation_report(
        results,
        model_name="Popularity",
        ks=KS,
    )


if __name__ == "__main__":
    main()
