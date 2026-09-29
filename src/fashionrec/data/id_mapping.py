from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IDMapping:
    user_to_idx: dict[str, int]
    idx_to_user: list[str]

    item_to_idx: dict[str, int]
    idx_to_item: list[str]

    @property
    def num_users(self) -> int:
        return len(self.idx_to_user)

    @property
    def num_items(self) -> int:
        return len(self.idx_to_item)

    def encode_user(self, user_id: str) -> int:
        return self.user_to_idx[str(user_id)]

    def encode_item(self, item_id: str) -> int:
        return self.item_to_idx[str(item_id)]

    def decode_user(self, user_idx: int) -> str:
        return self.idx_to_user[user_idx]

    def decode_item(self, item_idx: int) -> str:
        return self.idx_to_item[item_idx]


def build_id_mapping(
    train: pd.DataFrame,
) -> IDMapping:
    """
    Build integer ID mappings using TRAIN data only.

    Validation/test cold items are intentionally not added.
    """

    users = sorted(
        train["user_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    items = sorted(
        train["item_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(users)
    }

    item_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(items)
    }

    return IDMapping(
        user_to_idx=user_to_idx,
        idx_to_user=users,
        item_to_idx=item_to_idx,
        idx_to_item=items,
    )
