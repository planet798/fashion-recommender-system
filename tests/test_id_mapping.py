import pandas as pd
import pytest

from fashionrec.data.id_mapping import (
    build_id_mapping,
)


def test_build_id_mapping():
    train = pd.DataFrame(
        {
            "user_id": ["u2", "u1", "u2"],
            "item_id": ["b", "a", "c"],
        }
    )

    mapping = build_id_mapping(train)

    assert mapping.num_users == 2
    assert mapping.num_items == 3

    user_idx = mapping.encode_user("u1")
    item_idx = mapping.encode_item("b")

    assert mapping.decode_user(user_idx) == "u1"
    assert mapping.decode_item(item_idx) == "b"


def test_unknown_item_is_not_encoded():
    train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["a"],
        }
    )

    mapping = build_id_mapping(train)

    with pytest.raises(KeyError):
        mapping.encode_item("cold_item")
