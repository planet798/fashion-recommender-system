from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from fashionrec.data.id_mapping import (
    build_id_mapping,
)
from fashionrec.models.lightgcn import (
    LightGCN,
    build_normalized_adjacency,
)
from fashionrec.training.negative_sampling import (
    BPRNegativeSampler,
)


DATASETS = {
    "dev": Path("data/processed/dev_split"),
    "full": Path("data/processed/collab_5core"),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["dev", "full"],
        default="dev",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    torch.manual_seed(args.seed)

    device = get_device(args.device)

    root = DATASETS[args.dataset]

    train_path = root / "train.parquet"

    print("=" * 60)
    print("FashionRec V2 - Scalable LightGCN Training")
    print("=" * 60)

    print(f"Dataset       : {args.dataset}")
    print(f"Device        : {device}")
    print(f"Embedding dim : {args.embedding_dim}")
    print(f"Layers        : {args.num_layers}")
    print(f"Epochs        : {args.epochs}")
    print(f"BPR batch     : {args.batch_size:,}")
    print(f"Learning rate : {args.lr}")
    print()

    print("Loading training data...")

    train = pd.read_parquet(
        train_path,
        columns=[
            "user_id",
            "item_id",
        ],
    )

    mapping = build_id_mapping(train)

    user_indices = torch.tensor(
        train["user_id"]
        .astype(str)
        .map(mapping.user_to_idx)
        .to_numpy(),
        dtype=torch.long,
    )

    item_indices = torch.tensor(
        train["item_id"]
        .astype(str)
        .map(mapping.item_to_idx)
        .to_numpy(),
        dtype=torch.long,
    )

    num_interactions = len(train)

    print(f"Users        : {mapping.num_users:,}")
    print(f"Items        : {mapping.num_items:,}")
    print(f"Interactions : {num_interactions:,}")

    print()
    print("Building sparse graph...")

    adjacency = build_normalized_adjacency(
        user_indices,
        item_indices,
        num_users=mapping.num_users,
        num_items=mapping.num_items,
    ).to(device)

    model = LightGCN(
        num_users=mapping.num_users,
        num_items=mapping.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    sampler = BPRNegativeSampler(
        user_indices=user_indices,
        item_indices=item_indices,
        num_users=mapping.num_users,
        num_items=mapping.num_items,
        seed=args.seed,
    )

    print()
    print("Starting training...")
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        model.train()

        # One negative for every positive interaction.
        negative_indices = sampler.sample(
            user_indices
        )

        # Shuffle training triples every epoch.
        permutation = torch.randperm(
            num_interactions
        )

        shuffled_users = user_indices[
            permutation
        ]

        shuffled_positive = item_indices[
            permutation
        ]

        shuffled_negative = negative_indices[
            permutation
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        # Full-graph propagation ONCE per epoch.
        (
            user_embeddings,
            item_embeddings,
        ) = model(adjacency)

        epoch_loss = 0.0

        num_batches = (
            num_interactions
            + args.batch_size
            - 1
        ) // args.batch_size

        for batch_idx, start in enumerate(
            range(
                0,
                num_interactions,
                args.batch_size,
            )
        ):
            end = min(
                start + args.batch_size,
                num_interactions,
            )

            batch_users = shuffled_users[
                start:end
            ].to(
                device,
                non_blocking=True,
            )

            batch_positive = shuffled_positive[
                start:end
            ].to(
                device,
                non_blocking=True,
            )

            batch_negative = shuffled_negative[
                start:end
            ].to(
                device,
                non_blocking=True,
            )

            loss = model.bpr_batch_loss(
                users=batch_users,
                positive_items=batch_positive,
                negative_items=batch_negative,
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                reg_weight=args.reg_weight,
            )

            # Weight each batch so accumulated gradients
            # equal an interaction-level mean.
            batch_weight = (
                (end - start)
                / num_interactions
            )

            weighted_loss = (
                loss * batch_weight
            )

            retain_graph = (
                batch_idx
                < num_batches - 1
            )

            weighted_loss.backward(
                retain_graph=retain_graph
            )

            epoch_loss += (
                loss.item()
                * (end - start)
            )

        optimizer.step()

        epoch_loss /= num_interactions

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{args.epochs} "
            f"| loss={epoch_loss:.6f} "
            f"| batches={num_batches:,} "
            f"| time={elapsed:.2f}s"
        )

    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"lightgcn_{args.dataset}.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "num_users":
                mapping.num_users,

            "num_items":
                mapping.num_items,

            "embedding_dim":
                args.embedding_dim,

            "num_layers":
                args.num_layers,

            "seed":
                args.seed,

            "dataset":
                args.dataset,
        },
        checkpoint_path,
    )

    print()
    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()
