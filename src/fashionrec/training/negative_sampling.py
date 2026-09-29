from __future__ import annotations

import numpy as np
import torch


class BPRNegativeSampler:
    """
    Sample items that a user has never interacted with.

    One negative item is sampled for every positive interaction.
    """

    def __init__(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        num_users: int,
        num_items: int,
        seed: int = 42,
    ):
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

        self.positive_items = [
            set()
            for _ in range(num_users)
        ]

        for user, item in zip(
            user_indices.tolist(),
            item_indices.tolist(),
        ):
            self.positive_items[user].add(item)

    def sample(
        self,
        users: torch.Tensor,
    ) -> torch.Tensor:
        users_np = (
            users.detach()
            .cpu()
            .numpy()
        )

        negatives = np.empty(
            len(users_np),
            dtype=np.int64,
        )

        for index, user in enumerate(users_np):
            positives = self.positive_items[
                int(user)
            ]

            if len(positives) >= self.num_items:
                raise RuntimeError(
                    f"User {user} has interacted "
                    "with every item."
                )

            while True:
                candidate = int(
                    self.rng.integers(
                        self.num_items
                    )
                )

                if candidate not in positives:
                    negatives[index] = candidate
                    break

        return torch.from_numpy(negatives)
