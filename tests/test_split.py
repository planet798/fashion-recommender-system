import pandas as pd

from fashionrec.data.split import leave_last_out_split


def test_leave_last_out_split():
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u1", "u1", "u1", "u1",
                "u2", "u2", "u2", "u2",
            ],
            "item_id": [
                "a", "b", "c", "d",
                "e", "f", "g", "h",
            ],
            "timestamp": [
                1, 2, 3, 4,
                1, 2, 3, 4,
            ],
        }
    )

    train, validation, test = leave_last_out_split(
        interactions
    )

    assert train["item_id"].tolist() == [
        "a", "b",
        "e", "f",
    ]

    assert validation["item_id"].tolist() == [
        "c",
        "g",
    ]

    assert test["item_id"].tolist() == [
        "d",
        "h",
    ]


def test_remove_users_with_too_few_interactions():
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u1", "u1",
                "u2", "u2", "u2",
            ],
            "item_id": [
                "a", "b",
                "c", "d", "e",
            ],
            "timestamp": [
                1, 2,
                1, 2, 3,
            ],
        }
    )

    train, validation, test = leave_last_out_split(
        interactions,
        min_interactions=3,
    )

    assert set(train["user_id"]) == {"u2"}
    assert set(validation["user_id"]) == {"u2"}
    assert set(test["user_id"]) == {"u2"}
