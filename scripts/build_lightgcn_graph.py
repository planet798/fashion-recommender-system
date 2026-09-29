from pathlib import Path

import pandas as pd
import torch

from fashionrec.data.id_mapping import (
    build_id_mapping,
)
from fashionrec.models.lightgcn import (
    build_normalized_adjacency,
)


ROOT = Path("data/processed/dev_split")


def main():
    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    mapping = build_id_mapping(train)

    users = torch.tensor(
        [
            mapping.encode_user(user_id)
            for user_id in train["user_id"]
        ],
        dtype=torch.long,
    )

    items = torch.tensor(
        [
            mapping.encode_item(item_id)
            for item_id in train["item_id"]
        ],
        dtype=torch.long,
    )

    adjacency = build_normalized_adjacency(
        users,
        items,
        num_users=mapping.num_users,
        num_items=mapping.num_items,
    )

    print("=" * 60)
    print("LightGCN Sparse Graph")
    print("=" * 60)

    print(
        f"Shape       : "
        f"{tuple(adjacency.shape)}"
    )

    print(
        f"Non-zero    : "
        f"{adjacency._nnz():,}"
    )

    print(
        f"Sparse      : "
        f"{adjacency.is_sparse}"
    )

    expected_edges = (
        2 * len(train)
    )

    print(
        f"Expected nnz: "
        f"{expected_edges:,}"
    )


if __name__ == "__main__":
    main()
