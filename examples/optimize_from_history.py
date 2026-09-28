from src.ragbench.benchmark.summary import (
    load_experiment_summaries,
)

from src.ragbench.evaluation.experiment_optimizer import (
    find_best_experiment,
)


def main():
    experiments = load_experiment_summaries()

    print(
        f"Loaded {len(experiments)} "
        "historical experiments.\n"
    )

    metrics = [
        "recall",
        "precision",
        "reciprocal_rank",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    for metric in metrics:

        best = find_best_experiment(
            experiments,
            metric=metric,
        )

        if best is None:
            continue

        metric_key = {
            "recall": "recall_at_k",
            "precision": "precision_at_k",
            "reciprocal_rank": "reciprocal_rank",
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "answer_correctness": "answer_correctness",
        }[metric]

        value = best["metrics"][metric_key]

        print(
            f"{metric:20s} → "
            f"{best['name']:15s} "
            f"({value:.4f})"
        )


if __name__ == "__main__":
    main()