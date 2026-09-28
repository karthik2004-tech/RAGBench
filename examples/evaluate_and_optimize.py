import json

from src.ragbench.benchmark.summary import (
    load_experiment_summaries,
)

from src.ragbench.evaluation.optimization_engine import (
    optimize_result,
)

from src.ragbench.evaluation.dataset_optimizer import (
    optimize_dataset,
)

from src.ragbench.adapters.v1_adapter import (
    V1RAGAdapter,
)

from src.ragbench.datasets.schema import (
    EvaluationSample,
)

from src.ragbench.evaluation.evaluator import (
    RAGEvaluator,
)


def load_evaluation_samples():
    with open(
        "data/eval_set/eval_set.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return [
        EvaluationSample(
            id=item["id"],
            question=item["question"],
            ground_truth_answer=item["ground_truth_answer"],
            reference_passages=item["reference_passages"],
        )
        for item in data
    ]


def main():

    print("=" * 70)
    print("RAGBench - Evaluate and Optimize")
    print("=" * 70)

    # --------------------------------------------------
    # Load historical experiments
    # --------------------------------------------------

    experiments = load_experiment_summaries()

    print(
        f"\nLoaded {len(experiments)} "
        "historical experiments."
    )

    # --------------------------------------------------
    # Create RAG system
    # --------------------------------------------------

    rag = V1RAGAdapter()

    evaluator = RAGEvaluator(rag)

    # --------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------

    samples = load_evaluation_samples()

    print(
        f"Loaded {len(samples)} evaluation samples."
    )

    # --------------------------------------------------
    # Evaluate complete dataset
    # --------------------------------------------------

    results = []

    for sample in samples:

        print(
            f"\nEvaluating {sample.id}: "
            f"{sample.question}"
        )

        result = evaluator.evaluate_sample(
            sample,
            top_k=3,
        )

        results.append(result)

        print(
            f"  Recall: "
            f"{result['retrieval']['recall_at_k']:.3f}"
        )

        print(
            f"  Precision: "
            f"{result['retrieval']['precision_at_k']:.3f}"
        )

        print(
            f"  Faithfulness: "
            f"{result['generation']['faithfulness']}"
        )

    # --------------------------------------------------
    # Dataset-level optimization
    # --------------------------------------------------

    dataset_report = optimize_dataset(
        results
    )

    print("\n" + "=" * 70)
    print("DATASET OPTIMIZATION SUMMARY")
    print("=" * 70)

    print(
        f"Total samples: "
        f"{dataset_report['total_samples']}"
    )

    print(
        f"Passed: "
        f"{dataset_report['passed_samples']}"
    )

    print(
        f"Failed: "
        f"{dataset_report['failed_samples']}"
    )

    print(
        f"Failure rate: "
        f"{dataset_report['failure_rate']:.3f}"
    )

    print("\nFailure patterns:")

    if not dataset_report["prioritized_failures"]:

        print("  No failures detected.")

    else:

        for failure in dataset_report[
            "prioritized_failures"
        ]:

            print(
                f"  - "
                f"{failure['failure_type']}: "
                f"{failure['failure_rate']:.3f}"
            )

    # --------------------------------------------------
    # Per-question optimization
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-QUESTION OPTIMIZATION")
    print("=" * 70)

    for result in results:

        report = optimize_result(
            result,
            experiments,
        )

        diagnostics = report[
            "diagnostics"
        ]

        print(
            f"\n{result['id']}: "
            f"{diagnostics['overall_status']}"
        )

        print(
            f"  Primary diagnosis: "
            f"{diagnostics['primary_diagnosis']}"
        )

        if diagnostics["failure_types"]:

            print("  Failures:")

            for failure in diagnostics[
                "failure_types"
            ]:

                print(
                    f"    - {failure}"
                )

        historical = report[
            "historical_recommendations"
        ]

        available = [
            item
            for item in historical
            if item["available"]
        ]

        if available:

            print(
                "  Historical improvements:"
            )

            for recommendation in available:

                print(
                    f"    - "
                    f"{recommendation['metric']}: "
                    f"+{recommendation['improvement']:.3f} "
                    f"using "
                    f"{recommendation['recommended_experiment']['name']}"
                )


            next_experiment = report.get(
            "next_experiment"
        )

        if next_experiment:

            experiment = next_experiment["experiment"]

            print(
                "\n  Next experiment:"
            )

            print(
                f"    - "
                f"{experiment['name']}"
            )

            print(
                f"    - Target metric: "
                f"{next_experiment['target_metric']}"
            )

            print(
                f"    - Reason: "
                f"{next_experiment['reason']}"
            )


if __name__ == "__main__":
    main()