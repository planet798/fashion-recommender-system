from pathlib import Path

import pandas as pd

from fashionrec.data.split import (
    leave_last_out_split,
    validate_temporal_split,
)


INPUT = Path(
    "data/processed/interactions_dev.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/dev_split"
)


def main():
    print(f"Loading: {INPUT}")

    df = pd.read_parquet(INPUT)

    train, validation, test = leave_last_out_split(
        df,
        min_interactions=5,
    )

    validate_temporal_split(
        train,
        validation,
        test,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        OUTPUT_DIR / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        OUTPUT_DIR / "test.parquet",
        index=False,
    )

    print()
    print("=" * 60)
    print("Leave-Last-Out Split")
    print("=" * 60)

    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(validation):,}")
    print(f"Test       : {len(test):,}")

    print()
    print(f"Users      : {train['user_id'].nunique():,}")
    print(f"Train items: {train['item_id'].nunique():,}")
    print(f"Val items  : {validation['item_id'].nunique():,}")
    print(f"Test items : {test['item_id'].nunique():,}")

    print()
    print("Split validation: OK")


if __name__ == "__main__":
    main()
