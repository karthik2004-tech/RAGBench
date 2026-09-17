from src.ragbench.benchmark import ExperimentComparator


def create_experiment(
    name,
    recall,
    precision,
    rr,
    relevancy,
    faithfulness,
    correctness,
    latency,
):
    return {
        "experiment": {
            "name": name,
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 3,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "qwen2.5:1.5b",
        },
        "results": [
            {
                "id": "q001",
                "retrieval": {
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "reciprocal_rank": rr,
                },
                "generation": {
                    "answer_relevancy": relevancy,
                    "faithfulness": {
                        "score": faithfulness
                    },
                    "answer_correctness": correctness,
                },
                "latency_seconds": latency,
            }
        ],
    }


def test_compare_two_experiments():

    baseline = create_experiment(
        name="baseline",
        recall=0.8,
        precision=0.7,
        rr=1.0,
        relevancy=0.6,
        faithfulness=0.5,
        correctness=0.7,
        latency=1.0,
    )

    small_chunks = create_experiment(
        name="small_chunks",
        recall=0.6,
        precision=0.9,
        rr=0.5,
        relevancy=0.9,
        faithfulness=0.8,
        correctness=0.9,
        latency=2.0,
    )
    

    comparator = ExperimentComparator()

    comparison = comparator.compare(
        [
            baseline,
            small_chunks,
        ]
    )

    assert len(comparison["experiments"]) == 2

    assert comparison["best"]["recall"] == "baseline"

    assert comparison["best"]["precision"] == "small_chunks"

    assert comparison["best"]["reciprocal_rank"] == "baseline"

    assert comparison["best"]["answer_relevancy"] == "small_chunks"

    assert comparison["best"]["faithfulness"] == "small_chunks"

    assert comparison["best"]["answer_correctness"] == "small_chunks"

    assert comparison["best"]["latency_seconds"] == "baseline"


def test_compare_empty_experiments():

    comparator = ExperimentComparator()

    comparison = comparator.compare([])

    assert comparison["experiments"] == []

    assert comparison["best"] == {}


def test_ignore_experiment_without_results():

    empty_experiment = {
        "experiment": {
            "name": "empty",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 3,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "qwen2.5:1.5b",
        },
        "results": [],
    }

    valid_experiment = create_experiment(
        name="valid",
        recall=1.0,
        precision=1.0,
        rr=1.0,
        relevancy=1.0,
        faithfulness=1.0,
        correctness=1.0,
        latency=1.0,
    )

    comparator = ExperimentComparator()

    comparison = comparator.compare(
        [
            empty_experiment,
            valid_experiment,
        ]
    )

    assert len(comparison["experiments"]) == 1

    assert comparison["experiments"][0]["name"] == "valid"

    assert comparison["best"]["recall"] == "valid"