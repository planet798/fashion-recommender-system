from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math

import pandas as pd


class ItemCFRecommender:
    """
    Item-based Collaborative Filtering using
    user-item co-occurrence and cosine normalization.
    """

    def __init__(
        self,
        top_neighbors: int = 100,
    ):
        self.top_neighbors = top_neighbors

        self.item_counts: dict[str, int] = {}

        self.neighbors: dict[
            str,
            list[tuple[str, float]],
        ] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
    ) -> "ItemCFRecommender":

        print("Building ItemCF...")

        self.item_counts = (
            interactions[item_col]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        cooccurrence = defaultdict(Counter)

        histories = (
            interactions
            .groupby(user_col)[item_col]
            .apply(list)
        )

        for index, items in enumerate(
            histories,
            start=1,
        ):
            items = [str(item) for item in items]

            for item_i, item_j in combinations(items, 2):
                cooccurrence[item_i][item_j] += 1
                cooccurrence[item_j][item_i] += 1

            if index % 10000 == 0:
                print(
                    f"Processed users: "
                    f"{index:,}/{len(histories):,}"
                )

        print("Computing similarities...")

        for item_i, related in cooccurrence.items():

            similarities = []

            count_i = self.item_counts[item_i]

            for item_j, co_count in related.items():

                count_j = self.item_counts[item_j]

                similarity = (
                    co_count
                    / math.sqrt(
                        count_i * count_j
                    )
                )

                similarities.append(
                    (item_j, similarity)
                )

            similarities.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            self.neighbors[item_i] = (
                similarities[
                    : self.top_neighbors
                ]
            )

        print(
            f"Items with neighbors: "
            f"{len(self.neighbors):,}"
        )

        return self

    def recommend(
        self,
        history: set[str],
        k: int = 10,
    ) -> list[str]:

        scores = defaultdict(float)

        for history_item in history:

            for candidate, similarity in (
                self.neighbors.get(
                    str(history_item),
                    [],
                )
            ):

                if candidate in history:
                    continue

                scores[candidate] += similarity

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            item
            for item, _ in ranked[:k]
        ]
