from __future__ import annotations

import pandas as pd


def leave_last_out_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user-item interactions chronologically.

    For every eligible user:

        history[:-2] -> train
        history[-2]  -> validation
        history[-1]  -> test

    Users with fewer than `min_interactions` interactions are removed.
    """

    required_columns = {
        user_col,
        item_col,
        timestamp_col,
    }

    missing = required_columns - set(interactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if min_interactions < 3:
        raise ValueError(
            "min_interactions must be at least 3"
        )

    data = interactions.copy()

    # Remove invalid rows.
    data = data.dropna(
        subset=[
            user_col,
            item_col,
            timestamp_col,
        ]
    )

    # Only retain users with enough historical behaviour.
    user_counts = data.groupby(user_col)[item_col].transform("size")

    data = data[
        user_counts >= min_interactions
    ].copy()

    # Stable chronological sorting.
    data = data.sort_values(
        by=[user_col, timestamp_col],
        kind="stable",
    )

    # Latest interaction -> test.
    test_indices = (
        data.groupby(user_col, sort=False)
        .tail(1)
        .index
    )

    test = data.loc[test_indices]

    remaining = data.drop(index=test_indices)

    # Second latest interaction -> validation.
    validation_indices = (
        remaining.groupby(user_col, sort=False)
        .tail(1)
        .index
    )

    validation = remaining.loc[validation_indices]

    # Everything before validation/test -> train.
    train = remaining.drop(index=validation_indices)

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
