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

    positives = {
        0: {0, 1},
        1: {1, 2},
    }

    for _ in range(100):
        negatives = sampler.sample(
            sample_users
        )

        for user, negative in zip(
            sample_users.tolist(),
            negatives.tolist(),
        ):
            assert (
                negative
                not in positives[user]
            )


def test_negative_sampling_is_reproducible():
    users = torch.tensor(
        [0, 0, 1, 1]
    )

    items = torch.tensor(
        [0, 1, 1, 2]
    )

    sampler_a = BPRNegativeSampler(
        users,
        items,
        num_users=2,
        num_items=10,
        seed=42,
    )

    sampler_b = BPRNegativeSampler(
        users,
        items,
        num_users=2,
        num_items=10,
        seed=42,
    )

    query_users = torch.tensor(
        [0, 1, 0, 1]
    )

    result_a = sampler_a.sample(
        query_users
    )

    result_b = sampler_b.sample(
        query_users
    )

    assert torch.equal(
        result_a,
        result_b,
    )