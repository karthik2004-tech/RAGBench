from src.ragbench.benchmark import (
    ExperimentStorage,
    ExperimentComparator,
)


def main():

    storage = ExperimentStorage()

    # Automatically discover every saved experiment.
    experiments = storage.load_all()

    if not experiments:
        print("No experiment results found.")
        return

    comparator = ExperimentComparator()

    comparison = comparator.compare(
        experiments
    )

    print("\n" + "=" * 100)
    print("RAGBench Experiment Comparison")
    print("=" * 100)

    print(
        f"\n{'Experiment':<20}"
        f"{'Chunk':<10}"
        f"{'Overlap':<10}"
        f"{'Top-K':<8}"
        f"{'Recall':<10}"
        f"{'Precision':<10}"
        f"{'RR':<10}"
        f"{'Relevancy':<12}"
        f"{'Faithfulness':<14}"
        f"{'Correctness':<12}"
        f"{'Latency (s)':<16}"
    )

    print("-" * 110)

    for experiment in comparison["experiments"]:

        print(
            f"{experiment['name']:<20}"
            f"{experiment['chunk_size']:<10}"
            f"{experiment['chunk_overlap']:<10}"
            f"{experiment['top_k']:<8}"
            f"{experiment['recall']:<10.3f}"
            f"{experiment['precision']:<10.3f}"
            f"{experiment['reciprocal_rank']:<10.3f}"
            f"{experiment['answer_relevancy']:<12.3f}"
            f"{experiment['faithfulness']:<14.3f}"
            f"{experiment['answer_correctness']:<12.3f}"
            f"{experiment['avg_latency_seconds']:<16.3f}"
        )

    print("\n" + "=" * 100)
    print("Best Experiment by Metric")
    print("=" * 100)

    for metric, experiment_name in comparison["best"].items():

        print(
            f"{metric:<25}: {experiment_name}"
        )


if __name__ == "__main__":
    main()