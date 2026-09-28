from src.ragbench.evaluation.optimization_engine import (
    optimize_result,
)


def test_optimize_result():

    current_result = {
        "id": "q001",
        "question": "What is RAG?",

        "retrieval": {
            "recall_at_k": 0.4,
            "precision_at_k": 0.3,
            "reciprocal_rank": 0.4,
        },

        "generation": {
            "faithfulness": {
                "score": 0.2,
                "claims": [],
            },
            "answer_relevancy": 0.8,
            "answer_correctness": 0.5,
        },

        "latency_seconds": 1.0,
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
            "name": "topk_1",
            "metrics": {
                "recall_at_k": 0.76,
                "precision_at_k": 0.8,
                "reciprocal_rank": 0.8,
                "faithfulness": 0.2,
                "answer_relevancy": 0.8044,
                "answer_correctness": 0.77,
            },
            "config": {
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 1,
            },
        },
    ]

    report = optimize_result(
        current_result,
        experiments,
    )

    assert "diagnostics" in report

    assert "recommendations" in report

    assert "historical_recommendations" in report

    assert (
        report["diagnostics"]["overall_status"]
        == "fail"
    )

    assert len(
        report["recommendations"]
    ) > 0

    assert len(
        report["historical_recommendations"]
    ) > 0



def test_optimize_result_includes_ranking_recommendation():

    current_result = {
        "id": "q002",
        "question": "What are the main components of a RAG system?",

        "retrieval": {
            "recall_at_k": 1.0,
            "precision_at_k": 0.333,
            "reciprocal_rank": 0.333,
        },

        "generation": {
            "faithfulness": {
                "score": 1.0,
                "claims": [],
            },
            "answer_relevancy": 0.8,
            "answer_correctness": 0.4,
        },

        "latency_seconds": 1.0,
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

    report = optimize_result(
        current_result,
        experiments,
    )

    ranking_recommendations = [
        recommendation
        for recommendation in report["recommendations"]
        if recommendation["area"] == "ranking"
    ]

    assert len(ranking_recommendations) == 1
    assert ranking_recommendations[0]["priority"] == "high"


def test_optimize_result_recommends_next_experiment_for_ranking_failure():
    from src.ragbench.evaluation.optimization_engine import optimize_result

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

    result = optimize_result(current_result, experiments)

    assert result["diagnostics"]["primary_diagnosis"] == "ranking_failure"

    assert result["next_experiment"] is not None
    assert result["next_experiment"]["failure_type"] == "ranking_failure"
    assert result["next_experiment"]["target_metric"] == "reciprocal_rank"
    assert result["next_experiment"]["experiment"]["name"] == "chunk_800"


def test_optimize_result_produces_complete_optimization_report():
    from src.ragbench.evaluation.optimization_engine import optimize_result

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
        "latency_seconds": 1.0,
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

    report = optimize_result(
        current_result,
        experiments,
    )

    # Diagnosis
    assert report["diagnostics"]["overall_status"] == "fail"
    assert report["diagnostics"]["primary_diagnosis"] == "ranking_failure"

    # Recommendations
    assert report["recommendations"]

    # Historical evidence
    assert report["historical_recommendations"]

    # Next experiment
    next_experiment = report["next_experiment"]

    assert next_experiment is not None
    assert next_experiment["failure_type"] == "ranking_failure"
    assert next_experiment["target_metric"] == "reciprocal_rank"
    assert next_experiment["experiment"]["name"] == "chunk_800"