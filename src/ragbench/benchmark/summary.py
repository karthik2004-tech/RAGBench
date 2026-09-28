from typing import Any, Dict, List

from src.ragbench.benchmark.storage import ExperimentStorage


def extract_faithfulness(
    generation: Dict[str, Any],
) -> float:
    """
    Extract the numeric faithfulness score.

    Phase 2 stores faithfulness as a dictionary:

    {
        "score": 0.8,
        "claims": [...]
    }

    This function extracts the numeric score.
    """

    faithfulness = generation.get(
        "faithfulness",
        0.0,
    )

    if isinstance(faithfulness, dict):
        return float(
            faithfulness.get(
                "score",
                0.0,
            )
        )

    return float(faithfulness)


def calculate_average_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate average metrics across all questions
    in one experiment.
    """

    if not results:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": 0.0,
            "latency_seconds": 0.0,
        }

    totals = {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "reciprocal_rank": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "answer_correctness": 0.0,
        "latency_seconds": 0.0,
    }

    for result in results:

        retrieval = result.get(
            "retrieval",
            {},
        )

        generation = result.get(
            "generation",
            {},
        )

        totals["recall_at_k"] += float(
            retrieval.get(
                "recall_at_k",
                0.0,
            )
        )

        totals["precision_at_k"] += float(
            retrieval.get(
                "precision_at_k",
                0.0,
            )
        )

        totals["reciprocal_rank"] += float(
            retrieval.get(
                "reciprocal_rank",
                0.0,
            )
        )

        totals["faithfulness"] += (
            extract_faithfulness(
                generation
            )
        )

        totals["answer_relevancy"] += float(
            generation.get(
                "answer_relevancy",
                0.0,
            )
        )

        totals["answer_correctness"] += float(
            generation.get(
                "answer_correctness",
                0.0,
            )
        )

        totals["latency_seconds"] += float(
            result.get(
                "latency_seconds",
                0.0,
            )
        )

    count = len(results)

    return {
        metric: round(
            value / count,
            6,
        )
        for metric, value in totals.items()
    }


def summarize_experiment(
    experiment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert one stored Phase 2 experiment into
    a compact optimizer-ready summary.
    """

    experiment_config = experiment.get(
        "experiment",
        {},
    )

    results = experiment.get(
        "results",
        [],
    )

    metrics = calculate_average_metrics(
        results
    )

    return {
        "name": experiment_config.get(
            "name",
            "unknown",
        ),

        "config": {
            "chunk_size": experiment_config.get(
                "chunk_size"
            ),
            "chunk_overlap": experiment_config.get(
                "chunk_overlap"
            ),
            "top_k": experiment_config.get(
                "top_k"
            ),
            "embedding_model": experiment_config.get(
                "embedding_model"
            ),
            "llm_model": experiment_config.get(
                "llm_model"
            ),
        },

        "metrics": metrics,

        "num_samples": len(results),
    }


def summarize_experiments(
    experiments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert multiple stored experiments into
    optimizer-ready summaries.
    """

    return [
        summarize_experiment(
            experiment
        )
        for experiment in experiments
    ]


def load_experiment_summaries(
    output_dir: str = "data/experiments",
) -> List[Dict[str, Any]]:
    """
    Load all saved Phase 2 experiments and convert
    them into optimizer-ready summaries.
    """

    storage = ExperimentStorage(
        output_dir=output_dir
    )

    experiments = storage.load_all()

    return summarize_experiments(
        experiments
    )