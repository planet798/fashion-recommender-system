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
    Temporal leave-last-out split.

    For each user:

        history[:-2] -> train
        history[-2]  -> validation
        history[-1]  -> test
    """

    required = {
        user_col,
        item_col,
        timestamp_col,
    }

    missing = required - set(interactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if min_interactions < 3:
        raise ValueError(
            "min_interactions must be at least 3"
        )

    data = interactions.copy()

    data = data.dropna(
        subset=[
            user_col,
            item_col,
            timestamp_col,
        ]
    )

    counts = (
        data.groupby(user_col)[item_col]
        .transform("size")
    )

    data = data[
        counts >= min_interactions
    ].copy()

    data = data.sort_values(
        [user_col, timestamp_col],
        kind="stable",
    )

    rank_from_end = (
        data.groupby(user_col)
        .cumcount(ascending=False)
    )

    test = data[
        rank_from_end == 0
    ].copy()

    validation = data[
        rank_from_end == 1
    ].copy()

    train = data[
        rank_from_end >= 2
    ].copy()

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def validate_temporal_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate the main invariants of a leave-last-out split."""

    users = set(train["user_id"])

    if users != set(validation["user_id"]):
        raise ValueError(
            "Train and validation user sets do not match"
        )

    if users != set(test["user_id"]):
        raise ValueError(
            "Train and test user sets do not match"
        )

    if validation.groupby("user_id").size().max() != 1:
        raise ValueError(
            "Each user must have exactly one validation interaction"
        )

    if test.groupby("user_id").size().max() != 1:
        raise ValueError(
            "Each user must have exactly one test interaction"
        )

    train_last = train.groupby("user_id")["timestamp"].max()
    val_time = validation.set_index("user_id")["timestamp"]
    test_time = test.set_index("user_id")["timestamp"]

    if not (train_last <= val_time).all():
        raise ValueError(
            "Temporal leakage detected between train and validation"
        )

    if not (val_time <= test_time).all():
        raise ValueError(
            "Temporal leakage detected between validation and test"
        )