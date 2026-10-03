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
from fashionrec.evaluation.lightgcn_eval import (
    build_validation_sample,
    evaluate_lightgcn_topk,
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

    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval-users",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--loss-mode",
        choices=["full", "chunked"],
        default="chunked",
        help=(
            "full: compute BPR loss for all interactions at once; "
            "chunked: accumulate BPR gradients in batches"
    ),
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

    validation_path = (
        root / "validation.parquet"
    )

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
    print(f"Loss mode     : {args.loss_mode}")
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

    print()
    print("Preparing validation sample...")

    validation = pd.read_parquet(
        validation_path,
        columns=[
            "user_id",
            "item_id",
        ],
    )

    validation_sample = build_validation_sample(
        validation=validation,
        train_user_indices=user_indices,
        train_item_indices=item_indices,
        mapping=mapping,
        sample_size=args.eval_users,
        seed=args.seed,
    )

    print(
        f"Validation users: "
        f"{len(validation_sample.user_indices):,}"
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

    best_ndcg = -1.0
    best_epoch = 0
    patience_count = 0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        checkpoint_dir
        / (
            f"lightgcn_{args.dataset}"
            f"_dim{args.embedding_dim}"
            f"_l{args.num_layers}"
            f"_{args.loss_mode}"
            f"_best.pt"
        )
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

        optimizer.zero_grad(
            set_to_none=True
        )

        # Full graph propagation once per epoch.
        (
            user_embeddings,
            item_embeddings,
        ) = model(adjacency)


        if args.loss_mode == "full":
            # ----------------------------------
            # Full-Batch BPR
            # ----------------------------------

            training_users = user_indices.to(
                device,
                non_blocking=True,
            )

            positive_items = item_indices.to(
                device,
                non_blocking=True,
            )

            negative_items = negative_indices.to(
                device,
                non_blocking=True,
            )

            loss = model.bpr_batch_loss(
                users=training_users,
                positive_items=positive_items,
                negative_items=negative_items,
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                reg_weight=args.reg_weight,
            )

            loss.backward()

            optimizer.step()

            epoch_loss = loss.item()
            num_batches = 1


        else:
            # ----------------------------------
            # Chunked BPR
            # ----------------------------------

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

                batch_size_actual = (
                    end - start
                )

                batch_weight = (
                    batch_size_actual
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
                    * batch_size_actual
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

        should_evaluate = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if should_evaluate:
            model.eval()

            with torch.no_grad():
                (
                    eval_user_embeddings,
                    eval_item_embeddings,
                ) = model(adjacency)

            metrics = evaluate_lightgcn_topk(
                user_embeddings=eval_user_embeddings,
                item_embeddings=eval_item_embeddings,
                sample=validation_sample,
                k=10,
                batch_size=args.eval_batch_size,
            )

            current_ndcg = metrics["NDCG@10"]

            print(
                "Validation "
                f"| HR@10={metrics['HR@10']:.6f} "
                f"| NDCG@10={current_ndcg:.6f} "
                f"| MRR@10={metrics['MRR@10']:.6f}"
            )

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch
                patience_count = 0

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

                        "epoch":
                            epoch,

                        "validation_hr10":
                            metrics["HR@10"],

                        "validation_ndcg10":
                            current_ndcg,

                        "loss_mode":
                            args.loss_mode,
                    },
                    best_checkpoint_path,
                )

                print(
                    f"New best checkpoint saved: "
                    f"{best_checkpoint_path}"
                )

            else:
                patience_count += 1

                print(
                    f"No improvement: "
                    f"{patience_count}/{args.patience}"
                )

                if patience_count >= args.patience:
                    print()
                    print(
                        f"Early stopping at epoch {epoch}"
                    )
                    print(
                        f"Best epoch: {best_epoch}"
                    )
                    print(
                        f"Best NDCG@10: "
                        f"{best_ndcg:.6f}"
                    )

                    break

    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("Training finished")
    print("=" * 60)

    print(f"Best epoch   : {best_epoch}")
    print(f"Best NDCG@10 : {best_ndcg:.6f}")
    print(f"Best model   : {best_checkpoint_path}")


if __name__ == "__main__":
    main()
