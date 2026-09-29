from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """
    Recommend globally popular items.

    Used as the simplest non-personalized baseline.
    """

    def __init__(self):
        self.ranked_items: list[str] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        item_col: str = "item_id",
    ) -> "PopularityRecommender":

        counts = (
            interactions[item_col]
            .value_counts()
        )

        self.ranked_items = (
            counts.index
            .astype(str)
            .tolist()
        )

        return self

    def recommend(
        self,
        seen_items: set[str] | None = None,
        k: int = 10,
    ) -> list[str]:

        if not self.ranked_items:
            raise RuntimeError(
                "Model has not been fitted."
            )

        seen_items = seen_items or set()

        recommendations = []

        for item in self.ranked_items:
            if item in seen_items:
                continue

            recommendations.append(item)

            if len(recommendations) >= k:
                break

        return recommendations
