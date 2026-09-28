from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/dev_split")


def analyze(
    train: pd.DataFrame,
    target: pd.DataFrame,
    name: str,
):
    train_items = set(train["item_id"])
    target_items = set(target["item_id"])

    cold_items = target_items - train_items

    cold_rows = target[
        ~target["item_id"].isin(train_items)
    ]

    print(f"\n{name}")
    print("-" * 50)

    print(
        f"Unique items          : "
        f"{len(target_items):,}"
    )

    print(
        f"Cold unique items     : "
        f"{len(cold_items):,}"
    )

    print(
        f"Cold item rate        : "
        f"{len(cold_items) / len(target_items):.4%}"
    )

    print(
        f"Cold interactions     : "
        f"{len(cold_rows):,}"
    )

    print(
        f"Cold interaction rate : "
        f"{len(cold_rows) / len(target):.4%}"
    )


def main():
    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    validation = pd.read_parquet(
        ROOT / "validation.parquet"
    )

    test = pd.read_parquet(
        ROOT / "test.parquet"
    )

    print("=" * 60)
    print("FashionRec V2 - Cold Start Analysis")
    print("=" * 60)

    analyze(
        train,
        validation,
        "Validation",
    )

    analyze(
        train,
        test,
        "Test",
    )


if __name__ == "__main__":
    main()
