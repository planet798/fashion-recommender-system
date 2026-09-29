from pathlib import Path

import pandas as pd

from fashionrec.data.id_mapping import (
    build_id_mapping,
)


ROOT = Path("data/processed/dev_split")


def main():
    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    mapping = build_id_mapping(train)

    print("=" * 60)
    print("LightGCN Graph Statistics")
    print("=" * 60)

    print(f"Users        : {mapping.num_users:,}")
    print(f"Items        : {mapping.num_items:,}")
    print(f"Interactions : {len(train):,}")

    nodes = (
        mapping.num_users
        + mapping.num_items
    )

    print(f"Graph nodes  : {nodes:,}")

    density = (
        len(train)
        / (
            mapping.num_users
            * mapping.num_items
        )
    )

    print(f"Graph density: {density:.8%}")


if __name__ == "__main__":
    main()
