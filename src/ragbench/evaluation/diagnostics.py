from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DiagnosticConfig:
    recall_threshold: float = 0.5
    reciprocal_rank_threshold: float = 0.5
    faithfulness_threshold: float = 0.5
    relevancy_threshold: float = 0.5
    correctness_threshold: float = 0.5
    latency_threshold_seconds: float | None = None


def _extract_faithfulness_score(
    faithfulness: Any,
) -> float:
    """
    Convert faithfulness output into a numeric score.

    The evaluator may store faithfulness either as:

        0.8

    or:

        {
            "score": 0.8,
            "claims": [...]
        }
    """

    if isinstance(faithfulness, dict):
        return float(
            faithfulness.get("score", 0.0)
        )

    return float(faithfulness)


def diagnose_result(
    result: Dict[str, Any],
    config: DiagnosticConfig | None = None,
) -> Dict[str, Any]:

    if config is None:
        config = DiagnosticConfig()

    retrieval = result.get(
        "retrieval",
        {},
    )

    generation = result.get(
        "generation",
        {},
    )

    recall = float(
        retrieval.get(
            "recall_at_k",
            0.0,
        )
    )

    reciprocal_rank = float(
        retrieval.get(
            "reciprocal_rank",
            0.0,
        )
    )

    faithfulness = _extract_faithfulness_score(
        generation.get(
            "faithfulness",
            0.0,
        )
    )

    relevancy = float(
        generation.get(
            "answer_relevancy",
            0.0,
        )
    )

    correctness = float(
        generation.get(
            "answer_correctness",
            0.0,
        )
    )

    latency = result.get(
        "latency_seconds"
    )

    failures: List[str] = []
    explanations: List[str] = []

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieval_failed = (
        recall < config.recall_threshold
    )

    ranking_failed = (
        recall >= config.recall_threshold
        and reciprocal_rank
        < config.reciprocal_rank_threshold
    )

    if retrieval_failed:

        failures.append(
            "retrieval_failure"
        )

        explanations.append(
            "The retriever did not retrieve "
            "enough relevant evidence."
        )

    if ranking_failed:

        failures.append(
            "ranking_failure"
        )

        explanations.append(
            "Relevant evidence was retrieved, "
            "but it was ranked too low."
        )

    # --------------------------------------------------
    # Faithfulness
    # --------------------------------------------------

    if (
        faithfulness
        < config.faithfulness_threshold
    ):

        failures.append(
            "faithfulness_failure"
        )

        explanations.append(
            "The generated answer may not be "
            "sufficiently grounded in the "
            "retrieved context."
        )

    # --------------------------------------------------
    # Answer relevancy
    # --------------------------------------------------

    if (
        relevancy
        < config.relevancy_threshold
    ):

        failures.append(
            "answer_relevancy_failure"
        )

        explanations.append(
            "The generated answer may not "
            "sufficiently address the user's question."
        )

    # --------------------------------------------------
    # Answer correctness
    # --------------------------------------------------

    if (
        correctness
        < config.correctness_threshold
    ):

        failures.append(
            "answer_correctness_failure"
        )

        explanations.append(
            "The generated answer has low similarity "
            "to the expected ground-truth answer."
        )

    # --------------------------------------------------
    # Latency
    # --------------------------------------------------

    if (
        config.latency_threshold_seconds
        is not None
        and latency is not None
        and latency
        > config.latency_threshold_seconds
    ):

        failures.append(
            "latency_issue"
        )

        explanations.append(
            "The RAG pipeline exceeded the "
            "configured latency threshold."
        )

    # --------------------------------------------------
    # Final diagnosis
    # --------------------------------------------------

    if not failures:

        overall_status = "pass"

        primary_diagnosis = (
            "no_failure"
        )

        explanations.append(
            "The evaluation result did not cross "
            "any configured failure thresholds."
        )

    else:

        overall_status = "fail"

        priority = [
            "retrieval_failure",
            "ranking_failure",
            "faithfulness_failure",
            "answer_relevancy_failure",
            "answer_correctness_failure",
            "latency_issue",
        ]

        primary_diagnosis = next(
            failure
            for failure in priority
            if failure in failures
        )

    return {
        "overall_status": overall_status,
        "primary_diagnosis": primary_diagnosis,
        "failure_types": failures,
        "explanations": explanations,
    }