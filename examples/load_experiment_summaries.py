from src.ragbench.benchmark.summary import (
    load_experiment_summaries,
)


def main():

    summaries = load_experiment_summaries()

    print(
        f"Loaded {len(summaries)} experiments.\n"
    )

    for experiment in summaries:

        print("=" * 60)

        print(
            f"Experiment: {experiment['name']}"
        )

        print(
            "Configuration:"
        )

        print(
            f"  Chunk Size: "
            f"{experiment['config']['chunk_size']}"
        )

        print(
            f"  Chunk Overlap: "
            f"{experiment['config']['chunk_overlap']}"
        )

        print(
            f"  Top-K: "
            f"{experiment['config']['top_k']}"
        )

        print(
            "Metrics:"
        )

        for metric, value in experiment[
            "metrics"
        ].items():

            print(
                f"  {metric}: {value:.4f}"
            )


if __name__ == "__main__":
    main()