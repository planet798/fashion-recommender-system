from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/interactions_dev.parquet"
)


def main():
    df = pd.read_parquet(DATA_PATH)
    if df.empty:
        print("Dataset is empty.")
        return

    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    print("=" * 60)
    print("FashionRec V2 - Dataset EDA")
    print("=" * 60)

    print(f"Interactions : {len(df):,}")
    print(f"Users        : {df['user_id'].nunique():,}")
    print(f"Items        : {df['item_id'].nunique():,}")

    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()

    if num_users == 0 or num_items == 0:
        sparsity = 0.0
    else:
        sparsity = 1 - (
            len(df)
            / (num_users * num_items)
        )

    print(f"Sparsity     : {sparsity:.6%}")

    print("\nUser interactions")
    print(user_counts.describe())

    print("\nItem interactions")
    print(item_counts.describe())

    print("\nRating distribution")
    print(
        df["rating"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nTime range")
    print(df["datetime"].min())
    print(df["datetime"].max())

    print("\nTop 10 popular items")
    print(item_counts.nlargest(10))


if __name__ == "__main__":
    main()
