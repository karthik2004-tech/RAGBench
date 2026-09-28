from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RecommendationConfig:
    """
    Thresholds used to generate optimization recommendations.
    """

    low_recall_threshold: float = 0.5
    low_precision_threshold: float = 0.5
    low_reciprocal_rank_threshold: float = 0.5

    low_faithfulness_threshold: float = 0.5
    low_relevancy_threshold: float = 0.5
    low_correctness_threshold: float = 0.5

    high_latency_threshold_seconds: float | None = None


def generate_recommendations(
    result: Dict[str, Any],
    config: RecommendationConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate explainable optimization recommendations
    from a RAG evaluation result.

    Returns a list of recommendation dictionaries.
    """

    if config is None:
        config = RecommendationConfig()

    recommendations = []

    retrieval = result.get("retrieval", {})
    generation = result.get("generation", {})

    recall = retrieval.get("recall_at_k", 0.0)
    precision = retrieval.get("precision_at_k", 0.0)
    reciprocal_rank = retrieval.get("reciprocal_rank", 0.0)
    faithfulness = generation.get(
        "faithfulness",
        0.0,
    )

    if isinstance(faithfulness, dict):
        faithfulness = faithfulness.get(
        "score",
        0.0,
    )

    relevancy = generation.get("answer_relevancy", 0.0)
    correctness = generation.get("answer_correctness", 0.0)

    latency = result.get("latency_seconds")

    # ---------------------------------------------------------
    # Retrieval recommendations
    # ---------------------------------------------------------

    if recall < config.low_recall_threshold:

        recommendations.append(
            {
                "area": "retrieval",
                "priority": "high",
                "recommendation": (
                    "Experiment with higher Top-K values and "
                    "different chunking configurations to improve "
                    "the amount of relevant evidence retrieved."
                ),
                "reason": (
                    f"Recall@K is {recall:.3f}, which is below "
                    f"the configured threshold of "
                    f"{config.low_recall_threshold:.3f}."
                ),
            }
        )

    if precision < config.low_precision_threshold:

        recommendations.append(
            {
                "area": "retrieval",
                "priority": "medium",
                "recommendation": (
                    "Experiment with smaller or more focused chunks, "
                    "or consider adding a reranking stage to reduce "
                    "irrelevant retrieved context."
                ),
                "reason": (
                    f"Precision@K is {precision:.3f}, which is below "
                    f"the configured threshold of "
                    f"{config.low_precision_threshold:.3f}."
                ),
            }
        )

    if (
    recall >= config.low_recall_threshold
    and reciprocal_rank < config.low_reciprocal_rank_threshold
):

        recommendations.append(
            {
                "area": "ranking",
                "priority": "high",
                "recommendation": (
                    "Investigate retrieval ranking and consider "
                    "testing different embedding models, chunking "
                    "configurations, Top-K values, or a reranking "
                    "strategy to move relevant evidence higher in "
                    "the ranking."
                ),
                "reason": (
                    f"Recall@K is {recall:.3f}, indicating that relevant "
                    f"evidence was retrieved, but Reciprocal Rank is "
                    f"{reciprocal_rank:.3f}, indicating that the relevant "
                    f"evidence may be ranked too low."
                ),
            }
        )

    # ---------------------------------------------------------
    # Generation recommendations
    # ---------------------------------------------------------

    if faithfulness < config.low_faithfulness_threshold:

        recommendations.append(
            {
                "area": "generation",
                "priority": "high",
                "recommendation": (
                    "Investigate the generation prompt and model "
                    "grounding behavior. Encourage the model to "
                    "answer only from the retrieved context."
                ),
                "reason": (
                    f"Faithfulness is {faithfulness:.3f}, which is "
                    f"below the configured threshold of "
                    f"{config.low_faithfulness_threshold:.3f}."
                ),
            }
        )

    if relevancy < config.low_relevancy_threshold:

        recommendations.append(
            {
                "area": "generation",
                "priority": "medium",
                "recommendation": (
                    "Review the generation prompt and verify that "
                    "the retrieved context contains information "
                    "directly related to the question."
                ),
                "reason": (
                    f"Answer Relevancy is {relevancy:.3f}, which is "
                    f"below the configured threshold of "
                    f"{config.low_relevancy_threshold:.3f}."
                ),
            }
        )

    if correctness < config.low_correctness_threshold:

        recommendations.append(
            {
                "area": "generation",
                "priority": "medium",
                "recommendation": (
                    "Compare the generated answer with the "
                    "reference answer and investigate whether "
                    "retrieved evidence or generation behavior "
                    "is causing missing or incorrect information."
                ),
                "reason": (
                    f"Answer Correctness is {correctness:.3f}, "
                    f"which is below the configured threshold of "
                    f"{config.low_correctness_threshold:.3f}."
                ),
            }
        )

    # ---------------------------------------------------------
    # Latency recommendations
    # ---------------------------------------------------------

    if (
        config.high_latency_threshold_seconds is not None
        and latency is not None
        and latency > config.high_latency_threshold_seconds
    ):

        recommendations.append(
            {
                "area": "performance",
                "priority": "medium",
                "recommendation": (
                    "Investigate retrieval and generation latency. "
                    "Consider reducing Top-K, optimizing embedding "
                    "operations, or using a faster generation model."
                ),
                "reason": (
                    f"Latency is {latency:.3f} seconds, which exceeds "
                    f"the configured threshold of "
                    f"{config.high_latency_threshold_seconds:.3f} seconds."
                ),
            }
        )

    # ---------------------------------------------------------
    # No issues
    # ---------------------------------------------------------

    if not recommendations:

        recommendations.append(
            {
                "area": "overall",
                "priority": "info",
                "recommendation": (
                    "No optimization recommendation was triggered "
                    "by the configured thresholds."
                ),
                "reason": (
                    "All monitored metrics are at or above their "
                    "configured thresholds."
                ),
            }
        )

    return recommendations