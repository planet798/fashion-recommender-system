from fashionrec.evaluation.evaluator import (
    evaluate_rankings,
)


def test_evaluate_warm_and_cold_items():
    recommendations = {
        "u1": ["a", "b", "c"],
        "u2": ["x", "y", "z"],
    }

    ground_truth = {
        "u1": "b",
        "u2": "cold_item",
    }

    train_items = {
        "a",
        "b",
        "c",
        "x",
        "y",
        "z",
    }

    results = evaluate_rankings(
        recommendations,
        ground_truth,
        train_items,
        ks=(1, 3),
    )

    assert results["overall"]["users"] == 2
    assert results["warm"]["users"] == 1
    assert results["cold"]["users"] == 1

    assert results["warm"]["HR@1"] == 0.0
    assert results["warm"]["HR@3"] == 1.0

    assert results["cold"]["HR@3"] == 0.0


def test_missing_recommendations_are_allowed():
    recommendations = {}

    ground_truth = {
        "u1": "a",
    }

    train_items = {
        "a",
    }

    results = evaluate_rankings(
        recommendations,
        ground_truth,
        train_items,
        ks=(10,),
    )

    assert results["overall"]["HR@10"] == 0.0
    assert results["overall"]["non_empty_rate"] == 0.0
