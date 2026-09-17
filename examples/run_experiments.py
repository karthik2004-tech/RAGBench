import json

from src.ragbench.benchmark import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentStorage,
)

from src.ragbench.adapters import V1BenchmarkRAG
from src.ragbench.datasets.schema import EvaluationSample


def create_v1_rag(config: ExperimentConfig):
    return V1BenchmarkRAG(config)


def load_dataset():
    with open(
        "data/eval_set/eval_set.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return [
        EvaluationSample(**item)
        for item in data
    ]


def main():

    dataset = load_dataset()

    runner = ExperimentRunner(
        rag_factory=create_v1_rag,
        dataset=dataset,
    )
    storage = ExperimentStorage()

    
    experiments = [
        ExperimentConfig(
            name="baseline",
            chunk_size=400,
            chunk_overlap=50,
            top_k=3,
        ),
        ExperimentConfig(
            name="chunk_200",
            chunk_size=200,
            chunk_overlap=50,
            top_k=3,
        ),
        ExperimentConfig(
            name="chunk_600",
            chunk_size=600,
            chunk_overlap=50,
            top_k=3,
        ),
        ExperimentConfig(
            name="chunk_800",
            chunk_size=800,
            chunk_overlap=50,
            top_k=3,
        ),
        ExperimentConfig(
            name="overlap_0",
            chunk_size=400,
            chunk_overlap=0,
            top_k=3,
        ),
        ExperimentConfig(
            name="overlap_25",
            chunk_size=400,
            chunk_overlap=25,
            top_k=3,
        ),
        ExperimentConfig(
            name="overlap_100",
            chunk_size=400,
            chunk_overlap=100,
            top_k=3,
        ),
        ExperimentConfig(
            name="topk_1",
            chunk_size=400,
            chunk_overlap=50,
            top_k=1,
        ),
        ExperimentConfig(
            name="topk_2",
            chunk_size=400,
            chunk_overlap=50,
            top_k=2,
        ),
        ExperimentConfig(
            name="topk_5",
            chunk_size=400,
            chunk_overlap=50,
            top_k=5,
        ),
    ]


    for config in experiments:

        print("\n" + "=" * 60)
        print(f"Running experiment: {config.name}")
        print("=" * 60)

        result = runner.run(config)
        result_path = storage.save(result)

        print(
        f"\nResult saved to: {result_path}"
        )

        print(
            f"Chunk size: {config.chunk_size}"
        )

        print(
            f"Chunk overlap: {config.chunk_overlap}"
        )

        print(
            f"Top-K: {config.top_k}"
        )

        for evaluation in result["results"]:

            retrieval = evaluation["retrieval"]
            generation = evaluation["generation"]

            print(
                f"\nQuestion: {evaluation['question']}"
            )

            print(
                f"Recall@{config.top_k}: "
                f"{retrieval['recall_at_k']:.3f}"
            )

            print(
                f"Precision@{config.top_k}: "
                f"{retrieval['precision_at_k']:.3f}"
            )

            print(
                f"Reciprocal Rank: "
                f"{retrieval['reciprocal_rank']:.3f}"
            )

            print(
                f"Faithfulness: "
                f"{generation['faithfulness']['score']:.3f}"
            )

            print(
                f"Answer Relevancy: "
                f"{generation['answer_relevancy']:.3f}"
            )


if __name__ == "__main__":
    main()