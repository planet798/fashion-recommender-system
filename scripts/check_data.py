from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/interactions_dev.parquet")


def main():
    df = pd.read_parquet(DATA_PATH)

    print("=" * 60)
    print("FashionRec V2 - Data Quality Check")
    print("=" * 60)

    print(f"Interactions : {len(df):,}")
    print(f"Users        : {df['user_id'].nunique():,}")
    print(f"Items        : {df['item_id'].nunique():,}")

    duplicate_rows = df.duplicated().sum()

    duplicate_pairs = df.duplicated(
        subset=["user_id", "item_id"],
        keep=False,
    )

    duplicate_pair_rows = duplicate_pairs.sum()

    unique_pairs = (
        df[["user_id", "item_id"]]
        .drop_duplicates()
        .shape[0]
    )

    print()
    print(f"Exact duplicate rows       : {duplicate_rows:,}")
    print(f"Rows in repeated user-item : {duplicate_pair_rows:,}")
    print(f"Unique user-item pairs     : {unique_pairs:,}")

    pair_duplicate_rate = (
        1 - unique_pairs / len(df)
        if len(df)
        else 0
    )

    print(
        f"Repeated pair rate         : "
        f"{pair_duplicate_rate:.4%}"
    )

    print()
    print("Interactions per user after deduplication")

    dedup = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["user_id", "item_id"],
            keep="last",
        )
    )

    print(
        dedup.groupby("user_id")
        .size()
        .describe()
    )


if __name__ == "__main__":
    main()
