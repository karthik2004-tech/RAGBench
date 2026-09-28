from src.ragbench.benchmark.summary import (
    extract_faithfulness,
    calculate_average_metrics,
    summarize_experiment,
    summarize_experiments,
)


def test_extract_faithfulness_from_dict():

    generation = {
        "faithfulness": {
            "score": 0.8,
            "claims": [],
        }
    }

    score = extract_faithfulness(
        generation
    )

    assert score == 0.8


def test_extract_faithfulness_from_number():

    generation = {
        "faithfulness": 0.7
    }

    score = extract_faithfulness(
        generation
    )

    assert score == 0.7


def test_average_metrics():

    results = [
        {
            "retrieval": {
                "recall_at_k": 1.0,
                "precision_at_k": 0.5,
                "reciprocal_rank": 1.0,
            },
            "generation": {
                "faithfulness": {
                    "score": 0.8
                },
                "answer_relevancy": 0.9,
                "answer_correctness": 0.7,
            },
            "latency_seconds": 1.0,
        },
        {
            "retrieval": {
                "recall_at_k": 0.5,
                "precision_at_k": 0.3,
                "reciprocal_rank": 0.5,
            },
            "generation": {
                "faithfulness": {
                    "score": 0.6
                },
                "answer_relevancy": 0.7,
                "answer_correctness": 0.9,
            },
            "latency_seconds": 3.0,
        },
    ]

    metrics = calculate_average_metrics(
        results
    )

    assert metrics["recall_at_k"] == 0.75
    assert metrics["precision_at_k"] == 0.4
    assert metrics["reciprocal_rank"] == 0.75
    assert metrics["faithfulness"] == 0.7
    assert metrics["answer_relevancy"] == 0.8
    assert metrics["answer_correctness"] == 0.8
    assert metrics["latency_seconds"] == 2.0


def test_summarize_experiment():

    experiment = {
        "experiment": {
            "name": "topk_5",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 5,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "qwen2.5:1.5b",
        },

        "results": [
            {
                "retrieval": {
                    "recall_at_k": 1.0,
                    "precision_at_k": 0.5,
                    "reciprocal_rank": 1.0,
                },

                "generation": {
                    "faithfulness": {
                        "score": 0.5
                    },
                    "answer_relevancy": 0.8,
                    "answer_correctness": 0.9,
                },

                "latency_seconds": 1.2,
            }
        ],
    }

    summary = summarize_experiment(
        experiment
    )

    assert summary["name"] == "topk_5"

    assert (
        summary["config"]["chunk_size"]
        == 400
    )

    assert (
        summary["config"]["chunk_overlap"]
        == 50
    )

    assert (
        summary["config"]["top_k"]
        == 5
    )

    assert (
        summary["metrics"]["recall_at_k"]
        == 1.0
    )

    assert (
        summary["metrics"]["faithfulness"]
        == 0.5
    )

    assert (
        summary["num_samples"]
        == 1
    )


def test_summarize_multiple_experiments():

    experiments = [
        {
            "experiment": {
                "name": "experiment_a",
                "chunk_size": 400,
                "chunk_overlap": 50,
                "top_k": 3,
            },
            "results": [],
        },

        {
            "experiment": {
                "name": "experiment_b",
                "chunk_size": 200,
                "chunk_overlap": 50,
                "top_k": 5,
            },
            "results": [],
        },
    ]

    summaries = summarize_experiments(
        experiments
    )

    assert len(summaries) == 2

    assert (
        summaries[0]["name"]
        == "experiment_a"
    )

    assert (
        summaries[1]["name"]
        == "experiment_b"
    )