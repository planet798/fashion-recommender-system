import torch

from fashionrec.models.lightgcn import (
    LightGCN,
    build_normalized_adjacency,
)


def test_lightgcn_forward_shapes():
    # u0 -> i0, i1
    # u1 -> i1, i2

    users = torch.tensor(
        [0, 0, 1, 1]
    )

    items = torch.tensor(
        [0, 1, 1, 2]
    )

    adjacency = build_normalized_adjacency(
        users,
        items,
        num_users=2,
        num_items=3,
    )

    model = LightGCN(
        num_users=2,
        num_items=3,
        embedding_dim=8,
        num_layers=2,
    )

    user_embeddings, item_embeddings = (
        model(adjacency)
    )

    assert user_embeddings.shape == (
        2,
        8,
    )

    assert item_embeddings.shape == (
        3,
        8,
    )


def test_lightgcn_bpr_loss_is_finite():
    users = torch.tensor(
        [0, 0, 1, 1]
    )

    items = torch.tensor(
        [0, 1, 1, 2]
    )

    adjacency = build_normalized_adjacency(
        users,
        items,
        num_users=2,
        num_items=3,
    )

    model = LightGCN(
        num_users=2,
        num_items=3,
        embedding_dim=8,
        num_layers=1,
    )

    user_embeddings, item_embeddings = (
        model(adjacency)
    )

    batch_users = torch.tensor(
        [0, 1]
    )

    positive_items = torch.tensor(
        [0, 2]
    )

    negative_items = torch.tensor(
        [2, 0]
    )

    loss = model.bpr_loss(
        batch_users,
        positive_items,
        negative_items,
        user_embeddings,
        item_embeddings,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0
