from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_COLUMNS = {
    "user_id": "user_id",
    "parent_asin": "item_id",
    "rating": "rating",
    "timestamp": "timestamp",
}

def iterative_k_core(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Iteratively enforce minimum interaction counts for users and items.
    """

    data = df.copy()

    while True:
        previous_size = len(data)

        user_counts = data.groupby("user_id")["item_id"].transform("size")

        data = data[
            user_counts >= min_user_interactions
        ]

        item_counts = data.groupby("item_id")["user_id"].transform("size")

        data = data[
            item_counts >= min_item_interactions
        ]

        if len(data) == previous_size:
            break

    return data.reset_index(drop=True)


def preprocess_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize one Amazon Reviews interaction chunk."""

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()

    df = df.dropna(
        subset=["user_id", "item_id", "rating", "timestamp"]
    )

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(
        subset=["rating", "timestamp"]
    )

    df["timestamp"] = df["timestamp"].astype("int64")

    # Amazon timestamps are milliseconds.
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    return df.reset_index(drop=True)


def select_dev_users(
    df: pd.DataFrame,
    modulo: int = 100,
    bucket: int = 0,
) -> pd.DataFrame:
    """
    Deterministically sample users.

    modulo=100, bucket=0 keeps roughly 1% of users while
    preserving all interactions belonging to selected users.
    """

    hashes = pd.util.hash_pandas_object(
        df["user_id"],
        index=False,
    )

    mask = (hashes % modulo) == bucket

    return df.loc[mask].copy()


def build_dev_dataset(
    input_path: str | Path,
    output_path: str | Path,
    chunksize: int = 500_000,
    modulo: int = 100,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parts = []

    for chunk_id, chunk in enumerate(
        pd.read_csv(
            input_path,
            compression="gzip",
            chunksize=chunksize,
        ),
        start=1,
    ):
        clean = preprocess_chunk(chunk)
        sampled = select_dev_users(
            clean,
            modulo=modulo,
        )

        parts.append(sampled)

        print(
            f"chunk={chunk_id:03d} "
            f"raw={len(chunk):,} "
            f"dev={len(sampled):,}"
        )

    dev = pd.concat(
        parts,
        ignore_index=True,
    )

    print()
    print("Applying positive-feedback filter: rating >= 4")

    dev = dev[
        dev["rating"] >= 4.0
    ].copy()

    print(f"Positive interactions: {len(dev):,}")
    print()
    print("Filtering users with too few positive interactions")

    user_counts = dev.groupby("user_id")["item_id"].transform("size")

    dev = dev[
        user_counts >= 5
    ].copy()

    print(f"Interactions after user filtering: {len(dev):,}")

    dev = dev.sort_values(
        ["user_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    dev.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")
    print(f"Interactions: {len(dev):,}")
    print(f"Users: {dev['user_id'].nunique():,}")
    print(f"Items: {dev['item_id'].nunique():,}")


if __name__ == "__main__":
    build_dev_dataset(
        input_path="data/raw/Clothing_Shoes_and_Jewelry.csv.gz",
        output_path="data/processed/interactions_dev.parquet",
    )
