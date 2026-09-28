import argparse
from pathlib import Path

import pandas as pd

from fashionrec.data.split import leave_last_out_split


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input interaction CSV",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading interactions: {input_path}")

    interactions = pd.read_csv(input_path)

    print(
        f"Raw interactions: {len(interactions):,}"
    )

    train, validation, test = leave_last_out_split(
        interactions
    )

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test: {len(test):,}")

    train.to_parquet(
        output_dir / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        output_dir / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        output_dir / "test.parquet",
        index=False,
    )

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
