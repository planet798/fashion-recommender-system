from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from fashionrec.evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)


def evaluate_rankings(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, str],
    train_items: set[str],
    ks: Sequence[int] = (5, 10, 20),
) -> dict[str, dict[str, float]]:
    """
    Evaluate Top-K recommendation results.

    Scopes:
        overall: all users
        warm: target item appeared in training data
        cold: target item never appeared in training data
    """

    scopes = ("overall", "warm", "cold")

    totals = {
        scope: defaultdict(float)
        for scope in scopes
    }

    counts = {
        scope: 0
        for scope in scopes
    }

    non_empty = {
        scope: 0
        for scope in scopes
    }

    recommendation_lengths = {
        scope: 0
        for scope in scopes
    }

    for user_id, target in ground_truth.items():
        target = str(target)

        user_recommendations = [
            str(item)
            for item in recommendations.get(user_id, [])
        ]

        target_scope = (
            "warm"
            if target in train_items
            else "cold"
        )

        active_scopes = (
            "overall",
            target_scope,
        )

        for scope in active_scopes:
            counts[scope] += 1

            recommendation_lengths[scope] += len(
                user_recommendations
            )

            if user_recommendations:
                non_empty[scope] += 1

            for k in ks:
                totals[scope][f"HR@{k}"] += (
                    hit_rate_at_k(
                        user_recommendations,
                        target,
                        k,
                    )
                )

                totals[scope][f"NDCG@{k}"] += (
                    ndcg_at_k(
                        user_recommendations,
                        target,
                        k,
                    )
                )

                totals[scope][f"MRR@{k}"] += (
                    mrr_at_k(
                        user_recommendations,
                        target,
                        k,
                    )
                )

    results = {}

    for scope in scopes:
        count = counts[scope]

        scope_result = {
            "users": count,
            "non_empty_rate": (
                non_empty[scope] / count
                if count
                else 0.0
            ),
            "avg_recommendations": (
                recommendation_lengths[scope] / count
                if count
                else 0.0
            ),
        }

        for k in ks:
            for metric in ("HR", "NDCG", "MRR"):
                key = f"{metric}@{k}"

                scope_result[key] = (
                    totals[scope][key] / count
                    if count
                    else 0.0
                )

        results[scope] = scope_result

    return results


def print_evaluation_report(
    results: dict[str, dict[str, float]],
    model_name: str,
    ks: Sequence[int] = (5, 10, 20),
) -> None:
    print("=" * 60)
    print(f"{model_name} - Validation")
    print("=" * 60)

    for scope in ("overall", "warm", "cold"):
        result = results[scope]

        print()
        print(scope.capitalize())
        print("-" * 50)

        print(f"Users               : {result['users']:,}")
        print(
            f"Non-empty rec rate  : "
            f"{result['non_empty_rate']:.4%}"
        )
        print(
            f"Avg recommendations : "
            f"{result['avg_recommendations']:.2f}"
        )
        print()

        for k in ks:
            print(
                f"HR@{k:<2}   = "
                f"{result[f'HR@{k}']:.6f}"
            )
            print(
                f"NDCG@{k:<2} = "
                f"{result[f'NDCG@{k}']:.6f}"
            )
            print(
                f"MRR@{k:<2}  = "
                f"{result[f'MRR@{k}']:.6f}"
            )
            print()
