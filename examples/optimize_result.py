from src.ragbench.benchmark.summary import (
    load_experiment_summaries,
)

from src.ragbench.evaluation.optimization_engine import (
    optimize_result,
)


def main():

    experiments = load_experiment_summaries()

    current_result = {
        "id": "demo",
        "question": "What is RAG?",

        "retrieval": {
            "recall_at_k": 0.60,
            "precision_at_k": 0.35,
            "reciprocal_rank": 0.50,
        },

        "generation": {
            "faithfulness": {
                "score": 0.20,
                "claims": [],
            },
            "answer_relevancy": 0.70,
            "answer_correctness": 0.65,
        },

        "latency_seconds": 1.4,
    }

    report = optimize_result(
        current_result,
        experiments,
    )

    diagnostics = report[
        "diagnostics"
    ]

    print("=" * 70)
    print("RAGBench Optimization Report")
    print("=" * 70)

    print(
        f"\nStatus: "
        f"{diagnostics['overall_status']}"
    )

    print(
        f"Primary diagnosis: "
        f"{diagnostics['primary_diagnosis']}"
    )

    print("\nDetected failures:")

    for failure in diagnostics[
        "failure_types"
    ]:
        print(f"  - {failure}")

    print("\nGeneral recommendations:")

    for recommendation in report[
        "recommendations"
    ]:

        print(
            f"\n  [{recommendation['priority']}] "
            f"{recommendation['area']}"
        )

        print(
            f"  {recommendation['recommendation']}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Historical experiment recommendations"
    )

    print(
        "=" * 70
    )

    for recommendation in report[
    "historical_recommendations"
]:

        print(
            f"\nMetric: "
            f"{recommendation['metric']}"
        )

    if not recommendation["available"]:

        print(
            "  No historical experiment "
            "performed better than the current result."
        )

        print(
            f"  Reason: "
            f"{recommendation['reason']}"
        )

        

    experiment = recommendation[
        "recommended_experiment"
    ]

    print(
        f"  Current: "
        f"{recommendation['current_value']:.4f}"
    )

    print(
        f"  Historical: "
        f"{recommendation['historical_value']:.4f}"
    )

    print(
        f"  Improvement: "
        f"{recommendation['improvement']:.4f}"
    )

    print(
        f"  Experiment: "
        f"{experiment['name']}"
    )

    config = experiment[
        "config"
    ]

    print(
        f"  Configuration: "
        f"chunk={config['chunk_size']}, "
        f"overlap={config['chunk_overlap']}, "
        f"top_k={config['top_k']}"
    )


if __name__ == "__main__":
    main()