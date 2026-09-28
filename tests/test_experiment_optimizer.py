from src.ragbench.evaluation.experiment_optimizer import (
    find_best_experiment,
    compare_with_best_experiment,
    generate_experiment_recommendation,
)


def make_experiment(
    name,
    recall,
    precision=0.5,
    reciprocal_rank=0.5,
    faithfulness=0.5,
    answer_relevancy=0.5,
    answer_correctness=0.5,
):
    return {
        "name": name,
        "metrics": {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": reciprocal_rank,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
        },
    }


def test_find_best_experiment():

    experiments = [
        make_experiment(
            "experiment_a",
            recall=0.6,
        ),
        make_experiment(
            "experiment_b",
            recall=0.9,
        ),
        make_experiment(
            "experiment_c",
            recall=0.7,
        ),
    ]

    best = find_best_experiment(
        experiments,
        metric="recall",
    )

    assert best["name"] == "experiment_b"


def test_compare_current_with_best():

    experiments = [
        make_experiment(
            "experiment_a",
            recall=0.6,
        ),
        make_experiment(
            "experiment_b",
            recall=0.9,
        ),
    ]

    current = {
        "metrics": {
            "recall_at_k": 0.7,
        }
    }

    comparison = compare_with_best_experiment(
        current,
        experiments,
        metric="recall",
    )

    assert comparison["comparison_available"] is True
    assert comparison["current_value"] == 0.7
    assert comparison["best_historical_value"] == 0.9
    assert comparison["improvement"] == 0.2


def test_historical_recommendation():

    experiments = [
        make_experiment(
            "topk_3",
            recall=0.8,
        ),
        make_experiment(
            "topk_5",
            recall=1.0,
        ),
    ]

    current = {
        "metrics": {
            "recall_at_k": 0.8,
        }
    }

    recommendation = generate_experiment_recommendation(
        current,
        experiments,
        metric="recall",
    )

    assert recommendation is not None

    assert (
        recommendation["type"]
        == "historical_experiment"
    )

    assert (
        recommendation["experiment"]["name"]
        == "topk_5"
    )


def test_no_improvement_available():

    experiments = [
        make_experiment(
            "experiment_a",
            recall=0.7,
        ),
        make_experiment(
            "experiment_b",
            recall=0.8,
        ),
    ]

    current = {
        "metrics": {
            "recall_at_k": 0.9,
        }
    }

    recommendation = generate_experiment_recommendation(
        current,
        experiments,
        metric="recall",
    )

    assert recommendation is not None

    assert (
        "already at least as good"
        in recommendation["recommendation"]
    )


def test_no_historical_experiments():

    current = {
        "metrics": {
            "recall_at_k": 0.7,
        }
    }

    recommendation = generate_experiment_recommendation(
        current,
        [],
        metric="recall",
    )

    assert recommendation is None



def test_recommend_historical_configuration():
    from src.ragbench.evaluation.experiment_optimizer import (
        recommend_historical_configuration,
    )

    current_result = {
        "metrics": {
            "recall_at_k": 0.6,
        }
    }

    experiments = [
        {
            "name": "experiment_a",
            "metrics": {
                "recall_at_k": 0.8,
            },
            "config": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        },
        {
            "name": "experiment_b",
            "metrics": {
                "recall_at_k": 0.7,
            },
            "config": {
                "chunk_size": 200,
                "chunk_overlap": 50,
                "top_k": 5,
            },
        },
    ]

    recommendation = recommend_historical_configuration(
        current_result,
        experiments,
        metric="recall",
    )

    assert recommendation is not None
    assert recommendation["available"] is True
    assert recommendation["metric"] == "recall"
    assert recommendation["historical_value"] == 0.8
    assert (
        recommendation[
            "recommended_experiment"
        ]["name"]
        == "experiment_a"
    )


def test_no_historical_improvement():
    from src.ragbench.evaluation.experiment_optimizer import (
        recommend_historical_configuration,
    )

    current_result = {
        "metrics": {
            "recall_at_k": 0.9,
        }
    }

    experiments = [
        {
            "name": "experiment_a",
            "metrics": {
                "recall_at_k": 0.8,
            },
            "config": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        }
    ]

    recommendation = recommend_historical_configuration(
        current_result,
        experiments,
        metric="recall",
    )

    assert recommendation["available"] is False
    assert (
        recommendation[
            "recommended_experiment"
        ]
        is None
    )

def test_recommend_next_experiment_for_ranking_failure():

    from src.ragbench.evaluation.experiment_optimizer import (
        recommend_next_experiment,
    )

    current_result = {
        "retrieval": {
            "recall_at_k": 1.0,
            "precision_at_k": 0.333,
            "reciprocal_rank": 0.333,
        },
        "generation": {
            "faithfulness": 1.0,
            "answer_relevancy": 0.8,
            "answer_correctness": 0.4,
        },
    }

    experiments = [
        {
            "name": "baseline",
            "metrics": {
                "recall_at_k": 1.0,
                "precision_at_k": 0.4,
                "reciprocal_rank": 0.8667,
                "faithfulness": 0.0,
                "answer_relevancy": 0.7628,
                "answer_correctness": 0.7339,
            },
            "config": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        },
        {
            "name": "chunk_800",
            "metrics": {
                "recall_at_k": 1.0,
                "precision_at_k": 0.333,
                "reciprocal_rank": 0.9,
                "faithfulness": 0.0,
                "answer_relevancy": 0.718,
                "answer_correctness": 0.754,
            },
            "config": {
                "chunk_size": 800,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        },
    ]

    recommendation = recommend_next_experiment(
        current_result,
        experiments,
        failure_type="ranking_failure",
    )

    assert recommendation is not None
    assert recommendation["failure_type"] == "ranking_failure"
    assert recommendation["target_metric"] == "reciprocal_rank"
    assert recommendation["experiment"]["name"] == "chunk_800"
    assert recommendation["experiment"]["config"]["chunk_size"] == 800


def test_recommend_next_experiment_maps_failure_types():
    from src.ragbench.evaluation.experiment_optimizer import (
        recommend_next_experiment,
    )

    current_result = {
        "retrieval": {
            "recall_at_k": 0.2,
            "precision_at_k": 0.3,
            "reciprocal_rank": 0.2,
        },
        "generation": {
            "faithfulness": 0.2,
            "answer_relevancy": 0.3,
            "answer_correctness": 0.4,
        },
    }

    experiments = [
        {
            "name": "test_experiment",
            "metrics": {
                "recall_at_k": 0.9,
                "precision_at_k": 0.8,
                "reciprocal_rank": 0.9,
                "faithfulness": 0.9,
                "answer_relevancy": 0.9,
                "answer_correctness": 0.9,
            },
            "config": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        }
    ]

    expected_mappings = {
        "retrieval_failure": "recall",
        "ranking_failure": "reciprocal_rank",
        "faithfulness_failure": "faithfulness",
        "answer_relevancy_failure": "answer_relevancy",
        "answer_correctness_failure": "answer_correctness",
    }

    for failure_type, expected_metric in expected_mappings.items():

        recommendation = recommend_next_experiment(
            current_result,
            experiments,
            failure_type=failure_type,
        )

        assert recommendation is not None
        assert recommendation["failure_type"] == failure_type
        assert recommendation["target_metric"] == expected_metric
        assert recommendation["experiment"]["name"] == "test_experiment"    