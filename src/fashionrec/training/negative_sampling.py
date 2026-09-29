from __future__ import annotations

import numpy as np
import torch


class BPRNegativeSampler:
    """
    Vectorized uniform negative sampler.

    A negative item is an item that the user has not interacted
    with in the training set.
    """

    def __init__(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        num_users: int,
        num_items: int,
        seed: int = 42,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

        users = (
            user_indices.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

        items = (
            item_indices.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

        # Encode every observed (user, item) pair as one int64:
        #
        # key = user * num_items + item
        #
        # Sorting these keys lets us check sampled negatives
        # with vectorized np.searchsorted instead of Python loops.
        positive_keys = (
            users * np.int64(num_items)
            + items
        )

        self.positive_keys = np.unique(
            positive_keys
        )

    def _collision_mask(
        self,
        users: np.ndarray,
        items: np.ndarray,
    ) -> np.ndarray:
        candidate_keys = (
            users * np.int64(self.num_items)
            + items
        )

        positions = np.searchsorted(
            self.positive_keys,
            candidate_keys,
        )

        valid_position = (
            positions
            < len(self.positive_keys)
        )

        collisions = np.zeros(
            len(candidate_keys),
            dtype=bool,
        )

        collisions[valid_position] = (
            self.positive_keys[
                positions[valid_position]
            ]
            == candidate_keys[valid_position]
        )

        return collisions

    def sample(
        self,
        users: torch.Tensor,
    ) -> torch.Tensor:
        users_np = (
            users.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

        negatives = self.rng.integers(
            low=0,
            high=self.num_items,
            size=len(users_np),
            dtype=np.int64,
        )

        collisions = self._collision_mask(
            users_np,
            negatives,
        )

        # Usually very few collisions because the
        # user-item matrix is extremely sparse.
        while collisions.any():
            count = int(collisions.sum())

            negatives[collisions] = (
                self.rng.integers(
                    low=0,
                    high=self.num_items,
                    size=count,
                    dtype=np.int64,
                )
            )

            collision_indices = np.flatnonzero(
                collisions
            )

            remaining = self._collision_mask(
                users_np[collision_indices],
                negatives[collision_indices],
            )

            new_collisions = np.zeros_like(
                collisions
            )

            new_collisions[
                collision_indices[remaining]
            ] = True

            collisions = new_collisions

        return torch.from_numpy(
            negatives
        )