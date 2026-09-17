from typing import Any, Dict, List


class ExperimentComparator:
    """
    Compare multiple RAGBench experiment results.
    """

    def compare(
        self,
        experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare experiment results using aggregate metrics.
        """

        if not experiments:
            return {
                "experiments": [],
                "best": {},
            }

        summaries = []

        for experiment in experiments:

            config = experiment["experiment"]
            results = experiment["results"]

            if not results:
                continue

            recall_scores = []
            precision_scores = []
            rr_scores = []
            relevancy_scores = []
            faithfulness_scores = []
            correctness_scores = []
            latency_scores = []

            for result in results:

                retrieval = result["retrieval"]
                generation = result["generation"]

                recall_scores.append(
                    retrieval["recall_at_k"]
                )

                precision_scores.append(
                    retrieval["precision_at_k"]
                )

                rr_scores.append(
                    retrieval["reciprocal_rank"]
                )

                relevancy_scores.append(
                    generation["answer_relevancy"]
                )

                faithfulness_scores.append(
                    generation["faithfulness"]["score"]
                )

                correctness_scores.append(
                    generation["answer_correctness"]
                )

                latency_scores.append(
                    result["latency_seconds"]
                )

            summary = {
                "name": config["name"],
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "top_k": config["top_k"],
                "recall": sum(recall_scores) / len(recall_scores),
                "precision": sum(precision_scores) / len(precision_scores),
                "reciprocal_rank": sum(rr_scores) / len(rr_scores),
                "answer_relevancy": (
                    sum(relevancy_scores)
                    / len(relevancy_scores)
                ),
                "faithfulness": (
                    sum(faithfulness_scores)
                    / len(faithfulness_scores)
                ),
                "answer_correctness": (
                    sum(correctness_scores)
                    / len(correctness_scores)
                ),
                "avg_latency_seconds": sum(latency_scores) / len(latency_scores),
            }

            summaries.append(summary)

        if not summaries:
            return {
                "experiments": [],
                "best": {},
            }

        best = {
            "recall": max(
                summaries,
                key=lambda x: x["recall"],
            )["name"],

            "precision": max(
                summaries,
                key=lambda x: x["precision"],
            )["name"],

            "reciprocal_rank": max(
                summaries,
                key=lambda x: x["reciprocal_rank"],
            )["name"],

            "answer_relevancy": max(
                summaries,
                key=lambda x: x["answer_relevancy"],
            )["name"],

            "faithfulness": max(
                summaries,
                key=lambda x: x["faithfulness"],
            )["name"],
            "answer_correctness": max(
                summaries,
                key=lambda x: x["answer_correctness"]
            )["name"],
            "latency_seconds": min(
                summaries,
                key=lambda x: x["avg_latency_seconds"],
            )["name"],
        }

        return {
            "experiments": summaries,
            "best": best,
        }