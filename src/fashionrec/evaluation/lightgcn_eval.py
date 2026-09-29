from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from fashionrec.data.id_mapping import IDMapping


@dataclass
class LightGCNValidationSample:
    user_indices: np.ndarray
    target_item_indices: np.ndarray
    seen_items: dict[int, np.ndarray]


def build_validation_sample(
    validation: pd.DataFrame,
    train_user_indices: torch.Tensor,
    train_item_indices: torch.Tensor,
    mapping: IDMapping,
    sample_size: int = 10_000,
    seed: int = 42,
) -> LightGCNValidationSample:
    """
    Build a deterministic warm-item validation sample.

    Only validation items that exist in the training item
    vocabulary are kept.
    """

    frame = validation[
        ["user_id", "item_id"]
    ].copy()

    frame["user_idx"] = (
        frame["user_id"]
        .astype(str)
        .map(mapping.user_to_idx)
    )

    frame["item_idx"] = (
        frame["item_id"]
        .astype(str)
        .map(mapping.item_to_idx)
    )

    # Remove cold items and any unknown users.
    frame = frame.dropna(
        subset=["user_idx", "item_idx"]
    )

    if len(frame) > sample_size:
        frame = frame.sample(
            n=sample_size,
            random_state=seed,
        )

    frame = frame.sort_values(
        "user_idx"
    ).reset_index(drop=True)

    validation_users = (
        frame["user_idx"]
        .astype(np.int64)
        .to_numpy(copy=True)
    )

    validation_targets = (
        frame["item_idx"]
        .astype(np.int64)
        .to_numpy(copy=True)
    )

    # Select only the training histories needed by the
    # validation sample. Avoid building histories for
    # all ~1.7M users.
    train_users = (
        train_user_indices
        .cpu()
        .numpy()
    )

    train_items = (
        train_item_indices
        .cpu()
        .numpy()
    )

    selected_user_mask = np.zeros(
        mapping.num_users,
        dtype=bool,
    )

    selected_user_mask[
        validation_users
    ] = True

    selected_rows = (
        selected_user_mask[
            train_users
        ]
    )

    selected_train_users = (
        train_users[selected_rows]
    )

    selected_train_items = (
        train_items[selected_rows]
    )

    history_lists: dict[int, list[int]] = {
        int(user): []
        for user in validation_users
    }

    for user, item in zip(
        selected_train_users,
        selected_train_items,
    ):
        history_lists[int(user)].append(
            int(item)
        )

    seen_items = {
        user: np.asarray(
            items,
            dtype=np.int64,
        )
        for user, items
        in history_lists.items()
    }

    return LightGCNValidationSample(
        user_indices=validation_users,
        target_item_indices=validation_targets,
        seen_items=seen_items,
    )


@torch.no_grad()
def evaluate_lightgcn_topk(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    sample: LightGCNValidationSample,
    k: int = 10,
    batch_size: int = 128,
) -> dict[str, float]:
    """
    Exact all-item Top-K evaluation.
    """

    device = user_embeddings.device

    total_hits = 0
    total_ndcg = 0.0
    total_mrr = 0.0

    num_users = len(
        sample.user_indices
    )

    for start in range(
        0,
        num_users,
        batch_size,
    ):
        end = min(
            start + batch_size,
            num_users,
        )

        batch_users_np = (
            sample.user_indices[
                start:end
            ]
        )

        batch_targets_np = (
            sample.target_item_indices[
                start:end
            ]
        )

        batch_users = torch.from_numpy(
            batch_users_np
        ).to(
            device=device,
            dtype=torch.long,
        )

        batch_targets = torch.from_numpy(
            batch_targets_np
        ).to(
            device=device,
            dtype=torch.long,
        )

        scores = (
            user_embeddings[batch_users]
            @ item_embeddings.T
        )

        # Filter items already seen in training.
        mask_rows: list[int] = []
        mask_items: list[int] = []

        for row, user in enumerate(
            batch_users_np
        ):
            seen = sample.seen_items[
                int(user)
            ]

            mask_rows.extend(
                [row] * len(seen)
            )

            mask_items.extend(
                seen.tolist()
            )

        if mask_rows:
            rows_tensor = torch.tensor(
                mask_rows,
                dtype=torch.long,
                device=device,
            )

            items_tensor = torch.tensor(
                mask_items,
                dtype=torch.long,
                device=device,
            )

            scores[
                rows_tensor,
                items_tensor,
            ] = -torch.inf

        topk = torch.topk(
            scores,
            k=k,
            dim=1,
        ).indices

        matches = topk.eq(
            batch_targets.unsqueeze(1)
        )

        hits = matches.any(dim=1)

        total_hits += int(
            hits.sum().item()
        )

        hit_rows = torch.nonzero(
            hits,
            as_tuple=False,
        ).squeeze(1)

        if len(hit_rows) > 0:
            ranks = (
                matches[hit_rows]
                .float()
                .argmax(dim=1)
                + 1
            ).float()

            ndcg = (
                1.0
                / torch.log2(
                    ranks + 1.0
                )
            )

            mrr = 1.0 / ranks

            total_ndcg += float(
                ndcg.sum().item()
            )

            total_mrr += float(
                mrr.sum().item()
            )

    return {
        f"HR@{k}":
            total_hits / num_users,

        f"NDCG@{k}":
            total_ndcg / num_users,

        f"MRR@{k}":
            total_mrr / num_users,

        "users":
            num_users,
    }
