import pandas as pd

TRAIN_PATH = "data/processed/collab_5core/train.parquet"
VAL_PATH = "data/processed/collab_5core/validation.parquet"
OUTPUT_PATH = "data/processed/eval/validation_warm_10k_seed42.parquet"

SAMPLE_SIZE = 10_000
SEED = 42


def main():
    print("Loading validation...")
    validation = pd.read_parquet(
        VAL_PATH,
        columns=["user_id", "item_id"],
    )

    print("Loading train item vocabulary...")
    train_items = pd.read_parquet(
        TRAIN_PATH,
        columns=["item_id"],
    )["item_id"].unique()

    warm_validation = validation[
        validation["item_id"].isin(train_items)
    ].copy()

    print(f"Validation users : {len(validation):,}")
    print(f"Warm users       : {len(warm_validation):,}")

    sample = warm_validation.sample(
        n=SAMPLE_SIZE,
        random_state=SEED,
    ).reset_index(drop=True)

    sample.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Sample users     : {len(sample):,}")
    print(f"Saved to         : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
