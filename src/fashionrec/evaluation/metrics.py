from __future__ import annotations

import math


def hit_rate_at_k(
    recommended: list[str],
    target: str,
    k: int,
) -> float:
    return float(
        target in recommended[:k]
    )


def recall_at_k(
    recommended: list[str],
    target: str,
    k: int,
) -> float:
    """
    For one held-out relevant item,
    Recall@K is equivalent to HitRate@K.
    """

    return hit_rate_at_k(
        recommended,
        target,
        k,
    )


def ndcg_at_k(
    recommended: list[str],
    target: str,
    k: int,
) -> float:

    topk = recommended[:k]

    if target not in topk:
        return 0.0

    rank = topk.index(target) + 1

    return 1.0 / math.log2(rank + 1)


def mrr_at_k(
    recommended: list[str],
    target: str,
    k: int,
) -> float:

    topk = recommended[:k]

    if target not in topk:
        return 0.0

    rank = topk.index(target) + 1

    return 1.0 / rank
