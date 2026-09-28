from collections import Counter
from typing import Any, Dict, List


def analyze_failures(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze diagnostic results across an evaluation dataset.

    Args:
        results: List of evaluation results produced by RAGEvaluator.

    Returns:
        Summary containing:
        - total_samples
        - passed_samples
        - failed_samples
        - failure_rate
        - failure_counts
        - failed_questions
    """

    total_samples = len(results)

    passed_samples = 0
    failed_samples = 0

    failure_counter = Counter()
    failed_questions = []

    for result in results:

        diagnostics = result.get("diagnostics", {})

        status = diagnostics.get(
            "overall_status",
            "unknown",
        )

        failure_types = diagnostics.get(
            "failure_types",
            [],
        )

        if status == "pass":
            passed_samples += 1

        elif status == "fail":
            failed_samples += 1

            for failure_type in failure_types:
                failure_counter[failure_type] += 1

            failed_questions.append(
                {
                    "id": result.get("id"),
                    "question": result.get("question"),
                    "primary_diagnosis": diagnostics.get(
                        "primary_diagnosis"
                    ),
                    "failure_types": failure_types,
                }
            )

    if total_samples > 0:
        failure_rate = failed_samples / total_samples
    else:
        failure_rate = 0.0

    failure_rates_by_type = {
        failure_type: count / total_samples
        for failure_type, count in failure_counter.items()
    } if total_samples > 0 else {}

    return {
        "total_samples": total_samples,
        "passed_samples": passed_samples,
        "failed_samples": failed_samples,
        "failure_rate": failure_rate,
        "failure_counts": dict(failure_counter),
        "failure_rates_by_type": failure_rates_by_type,
        "failed_questions": failed_questions,
    }