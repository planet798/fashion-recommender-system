from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_normalized_adjacency(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """
    Build the normalized bipartite adjacency matrix:

        D^(-1/2) A D^(-1/2)

    User nodes:
        [0, num_users)

    Item nodes:
        [num_users, num_users + num_items)
    """

    user_indices = user_indices.long()
    item_indices = item_indices.long()

    item_nodes = item_indices + num_users

    # User -> Item
    src_forward = user_indices
    dst_forward = item_nodes

    # Item -> User
    src_backward = item_nodes
    dst_backward = user_indices

    src = torch.cat(
        [src_forward, src_backward]
    )

    dst = torch.cat(
        [dst_forward, dst_backward]
    )

    num_nodes = num_users + num_items

    degree = torch.zeros(
        num_nodes,
        dtype=torch.float32,
    )

    degree.scatter_add_(
        0,
        src,
        torch.ones_like(
            src,
            dtype=torch.float32,
        ),
    )

    degree_inv_sqrt = degree.pow(-0.5)

    degree_inv_sqrt[
        torch.isinf(degree_inv_sqrt)
    ] = 0.0

    values = (
        degree_inv_sqrt[src]
        * degree_inv_sqrt[dst]
    )

    indices = torch.stack(
        [src, dst],
        dim=0,
    )

    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        check_invariants=False,
    )

    return adjacency.coalesce()


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.user_embedding.weight
        )

        nn.init.xavier_uniform_(
            self.item_embedding.weight
        )

    def forward(
        self,
        normalized_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run LightGCN graph propagation.

        Final embedding is the mean of:
            layer 0
            layer 1
            ...
            layer K
        """

        all_embeddings = torch.cat(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
            ],
            dim=0,
        )

        layer_embeddings = [
            all_embeddings
        ]

        current = all_embeddings

        for _ in range(self.num_layers):
            current = torch.sparse.mm(
                normalized_adjacency,
                current,
            )

            layer_embeddings.append(
                current
            )

        final_embeddings = torch.stack(
            layer_embeddings,
            dim=0,
        ).mean(dim=0)

        user_embeddings = (
            final_embeddings[
                : self.num_users
            ]
        )

        item_embeddings = (
            final_embeddings[
                self.num_users :
            ]
        )

        return (
            user_embeddings,
            item_embeddings,
        )

    def bpr_batch_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> torch.Tensor:

        user_vectors = user_embeddings[users]
        positive_vectors = item_embeddings[positive_items]
        negative_vectors = item_embeddings[negative_items]

        positive_scores = (
            user_vectors * positive_vectors
        ).sum(dim=1)

        negative_scores = (
            user_vectors * negative_vectors
        ).sum(dim=1)

        ranking_loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

        raw_user = self.user_embedding(users)
        raw_positive = self.item_embedding(
            positive_items
        )
        raw_negative = self.item_embedding(
            negative_items
        )

        regularization = (
            raw_user.pow(2).sum()
            + raw_positive.pow(2).sum()
            + raw_negative.pow(2).sum()
        ) / users.shape[0]

        return (
            ranking_loss
            + reg_weight * regularization
        )

    def bpr_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> torch.Tensor:

        user_vectors = user_embeddings[
            users
        ]

        positive_vectors = item_embeddings[
            positive_items
        ]

        negative_vectors = item_embeddings[
            negative_items
        ]

        positive_scores = (
            user_vectors
            * positive_vectors
        ).sum(dim=1)

        negative_scores = (
            user_vectors
            * negative_vectors
        ).sum(dim=1)

        ranking_loss = -F.logsigmoid(
            positive_scores
            - negative_scores
        ).mean()

        raw_user = self.user_embedding(
            users
        )

        raw_positive = self.item_embedding(
            positive_items
        )

        raw_negative = self.item_embedding(
            negative_items
        )

        regularization = (
            raw_user.pow(2).sum()
            + raw_positive.pow(2).sum()
            + raw_negative.pow(2).sum()
        ) / users.shape[0]

        return (
            ranking_loss
            + reg_weight * regularization
        )
