from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from fashionrec.data.id_mapping import (
    build_id_mapping,
)
from fashionrec.evaluation.evaluator import (
    evaluate_rankings,
    print_evaluation_report,
)
from fashionrec.models.lightgcn import (
    LightGCN,
    build_normalized_adjacency,
)


ROOT = Path("data/processed/dev_split")
CHECKPOINT = Path("checkpoints/lightgcn_dev.pt")

KS = (5, 10, 20)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    return torch.device(name)


def main():
    args = parse_args()

    device = get_device(args.device)

    print("=" * 60)
    print("FashionRec V2 - LightGCN Evaluation")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Batch size : {args.batch_size}")
    print()

    train = pd.read_parquet(
        ROOT / "train.parquet"
    )

    validation = pd.read_parquet(
        ROOT / "validation.parquet"
    )

    mapping = build_id_mapping(train)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = LightGCN(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    train_user_indices = torch.tensor(
        train["user_id"]
        .astype(str)
        .map(mapping.user_to_idx)
        .to_numpy(),
        dtype=torch.long,
    )

    train_item_indices = torch.tensor(
        train["item_id"]
        .astype(str)
        .map(mapping.item_to_idx)
        .to_numpy(),
        dtype=torch.long,
    )

    print("Building graph...")

    adjacency = build_normalized_adjacency(
        train_user_indices,
        train_item_indices,
        num_users=mapping.num_users,
        num_items=mapping.num_items,
    ).to(device)

    print("Computing final embeddings...")

    with torch.no_grad():
        (
            user_embeddings,
            item_embeddings,
        ) = model(adjacency)

    # User history in encoded item IDs.
    history_indices = {}

    for user_id, group in train.groupby("user_id"):
        encoded = [
            mapping.item_to_idx[str(item)]
            for item in group["item_id"]
        ]

        history_indices[str(user_id)] = encoded

    validation_users = (
        validation["user_id"]
        .astype(str)
        .tolist()
    )

    recommendations = {}

    max_k = max(KS)

    print()
    print("Retrieving Top-K items...")

    with torch.no_grad():

        for start in range(
            0,
            len(validation_users),
            args.batch_size,
        ):
            batch_user_ids = validation_users[
                start : start + args.batch_size
            ]

            batch_user_indices = torch.tensor(
                [
                    mapping.user_to_idx[user_id]
                    for user_id in batch_user_ids
                ],
                dtype=torch.long,
                device=device,
            )

            batch_user_embeddings = (
                user_embeddings[
                    batch_user_indices
                ]
            )

            scores = (
                batch_user_embeddings
                @ item_embeddings.T
            )

            # Do not recommend already-seen items.
            for row_index, user_id in enumerate(
                batch_user_ids
            ):
                seen = history_indices[user_id]

                scores[
                    row_index,
                    seen,
                ] = -torch.inf

            topk_indices = torch.topk(
                scores,
                k=max_k,
                dim=1,
            ).indices.cpu()

            for row_index, user_id in enumerate(
                batch_user_ids
            ):
                recommendations[user_id] = [
                    mapping.decode_item(
                        int(item_idx)
                    )
                    for item_idx in (
                        topk_indices[
                            row_index
                        ].tolist()
                    )
                ]

            processed = min(
                start + args.batch_size,
                len(validation_users),
            )

            if (
                processed % 10000 < args.batch_size
                or processed == len(validation_users)
            ):
                print(
                    f"Users: "
                    f"{processed:,}/"
                    f"{len(validation_users):,}"
                )

    ground_truth = {
        str(row.user_id): str(row.item_id)
        for row in validation.itertuples(
            index=False
        )
    }

    train_items = set(
        train["item_id"].astype(str)
    )

    results = evaluate_rankings(
        recommendations,
        ground_truth,
        train_items,
        ks=KS,
    )

    print()

    print_evaluation_report(
        results,
        model_name="LightGCN",
        ks=KS,
    )


if __name__ == "__main__":
    main()
