from src.ragbench.benchmark.summary import (
    load_experiment_summaries,
)

from src.ragbench.evaluation.experiment_optimizer import (
    recommend_historical_configuration,
)


def main():

    experiments = load_experiment_summaries()

    print(
        f"Loaded {len(experiments)} "
        "historical experiments.\n"
    )

    # Simulated current result.
    #
    # In the next step this will come directly
    # from a real RAG evaluation.
    current_result = {
        "metrics": {
            "recall_at_k": 0.60,
            "precision_at_k": 0.35,
            "reciprocal_rank": 0.50,
            "faithfulness": 0.20,
            "answer_relevancy": 0.70,
            "answer_correctness": 0.65,
        }
    }

    metrics = [
        "recall",
        "precision",
        "reciprocal_rank",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    print("=" * 60)
    print("Historical Configuration Recommendations")
    print("=" * 60)

    for metric in metrics:

        recommendation = (
            recommend_historical_configuration(
                current_result,
                experiments,
                metric=metric,
            )
        )

        if recommendation is None:
            continue

        print(
            f"\nMetric: {metric}"
        )

        if not recommendation["available"]:

            print(
                "  No historical experiment "
                "improved this metric."
            )

            continue

        experiment = recommendation[
            "recommended_experiment"
        ]

        print(
            f"  Current value: "
            f"{recommendation['current_value']:.4f}"
        )

        print(
            f"  Historical value: "
            f"{recommendation['historical_value']:.4f}"
        )

        print(
            f"  Potential improvement: "
            f"{recommendation['improvement']:.4f}"
        )

        print(
            f"  Experiment: "
            f"{experiment['name']}"
        )

        print("  Configuration:")

        config = experiment["config"]

        print(
            f"    Chunk size: "
            f"{config['chunk_size']}"
        )

        print(
            f"    Chunk overlap: "
            f"{config['chunk_overlap']}"
        )

        print(
            f"    Top-K: "
            f"{config['top_k']}"
        )


if __name__ == "__main__":
    main()