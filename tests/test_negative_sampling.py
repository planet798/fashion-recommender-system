import torch

from fashionrec.training.negative_sampling import (
    BPRNegativeSampler,
)


def test_negative_samples_are_not_positive():
    users = torch.tensor(
        [0, 0, 1, 1]
    )

    items = torch.tensor(
        [0, 1, 1, 2]
    )

    sampler = BPRNegativeSampler(
        user_indices=users,
        item_indices=items,
        num_users=2,
        num_items=4,
        seed=42,
    )

    sample_users = torch.tensor(
        [0, 0, 0, 1, 1, 1]
    )

    negatives = sampler.sample(
        sample_users
    )

    positives = {
        0: {0, 1},
        1: {1, 2},
    }

    for user, negative in zip(
        sample_users.tolist(),
        negatives.tolist(),
    ):
        assert negative not in positives[user]
