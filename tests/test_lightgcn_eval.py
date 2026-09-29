import numpy as np
import torch

from fashionrec.evaluation.lightgcn_eval import (
    LightGCNValidationSample,
    evaluate_lightgcn_topk,
)


def test_lightgcn_topk_evaluation():
    # Two users, four items.
    user_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    item_embeddings = torch.tensor(
        [
            [1.0, 0.0],   # seen by user 0
            [0.9, 0.0],   # target for user 0
            [0.0, 1.0],   # seen by user 1
            [0.0, 0.9],   # target for user 1
        ]
    )

    sample = LightGCNValidationSample(
        user_indices=np.array(
            [0, 1],
            dtype=np.int64,
        ),
        target_item_indices=np.array(
            [1, 3],
            dtype=np.int64,
        ),
        seen_items={
            0: np.array([0]),
            1: np.array([2]),
        },
    )

    results = evaluate_lightgcn_topk(
        user_embeddings,
        item_embeddings,
        sample,
        k=1,
        batch_size=2,
    )

    assert results["HR@1"] == 1.0
    assert results["NDCG@1"] == 1.0
    assert results["MRR@1"] == 1.0
