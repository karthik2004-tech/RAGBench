import json

from src.ragbench.adapters.v1_adapter import V1RAGAdapter
from src.ragbench.datasets.schema import EvaluationSample
from src.ragbench.evaluation.evaluator import RAGEvaluator


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvaluationSample(
            id=item["id"],
            question=item["question"],
            ground_truth_answer=item["ground_truth_answer"],
            relevant_chunk_ids=item["relevant_chunk_ids"],
        )
        for item in data
    ]


def main():
    dataset = load_dataset(
        "data/eval_set/eval_set.json"
    )

    rag = V1RAGAdapter()

    evaluator = RAGEvaluator(rag)

    results = evaluator.evaluate(
        dataset,
        top_k=3,
    )

    print("\n===== RAGBench V2 Evaluation =====\n")

    for result in results:
        print(f"Question: {result['question']}")

        print(
            f"Recall@3: "
            f"{result['retrieval']['recall_at_k']:.3f}"
        )

        print(
            f"Precision@3: "
            f"{result['retrieval']['precision_at_k']:.3f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{result['retrieval']['reciprocal_rank']:.3f}"
        )

        print(
            f"Faithfulness: "
            f"{result['generation']['faithfulness']['score']:.3f}"
        )

        print(
            f"Answer Relevancy: "
            f"{result['generation']['answer_relevancy']:.3f}"
        )

        print("\nGenerated Answer:")
        print(result["generated_answer"])

        print("\nRetrieved Chunks:")

        for chunk in result["retrieved_chunks"]:
            print(
                f"- {chunk['chunk_id']} "
                f"(score={chunk['score']:.3f})"
            )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()