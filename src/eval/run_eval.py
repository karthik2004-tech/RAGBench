"""
Run the full evaluation harness: load eval set -> retrieve -> generate -> score.

Usage:
    python -m src.eval.run_eval --eval-set data/eval_set/eval_set.json --top-k 3
"""
import argparse
import json

from src.rag.retrieve import Retriever
from src.rag.generate import generate_answer
from src.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank
from src.eval.faithfulness import faithfulness_score
from src.eval.answer_relevancy import answer_relevancy_score


def run_eval(eval_set_path: str, top_k: int = 3, out_path: str = "data/eval_set/results.json"):
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    retriever = Retriever()
    results = []

    for item in eval_set:
        question = item["question"]
        ground_truth_chunk_id = item["ground_truth_chunk_id"]

        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        context = "\n\n".join(r["text"] for r in retrieved)

        answer = generate_answer(question, retrieved)

        result = {
            "id": item["id"],
            "question": question,
            "generated_answer": answer,
            "retrieved_chunk_ids": retrieved_ids,
            "ground_truth_chunk_id": ground_truth_chunk_id,
            "recall_at_k": recall_at_k(retrieved_ids, ground_truth_chunk_id, top_k),
            "precision_at_k": precision_at_k(retrieved_ids, ground_truth_chunk_id, top_k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, ground_truth_chunk_id),
            "faithfulness": faithfulness_score(answer, context),
            "answer_relevancy": answer_relevancy_score(question, answer),
        }
        results.append(result)
        print(f"[{item['id']}] recall@{top_k}={result['recall_at_k']} "
              f"faithfulness={result['faithfulness']['score']} "
              f"answer_relevancy={result['answer_relevancy']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # summary
    n = len(results)
    avg_recall = sum(r["recall_at_k"] for r in results) / n
    avg_faithfulness = sum(r["faithfulness"]["score"] for r in results) / n
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n

    print("\n--- Summary ---")
    print(f"Recall@{top_k}:      {avg_recall:.3f}")
    print(f"MRR:               {mrr:.3f}")
    print(f"Faithfulness:      {avg_faithfulness:.3f}")
    print(f"Answer Relevancy:  {avg_relevancy:.3f}")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval_set/eval_set.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default="data/eval_set/results.json")
    args = parser.parse_args()
    run_eval(args.eval_set, args.top_k, args.out)
