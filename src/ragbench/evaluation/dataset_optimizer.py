from typing import Any, Dict, List

from src.ragbench.evaluation.failure_analysis import (
    analyze_failures,
)


def optimize_dataset(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze a complete evaluation dataset and summarize
    recurring RAG failure patterns.
    """

    analysis = analyze_failures(results)

    failure_rates = analysis["failure_rates_by_type"]

    prioritized_failures = sorted(
        failure_rates.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "total_samples": analysis["total_samples"],
        "passed_samples": analysis["passed_samples"],
        "failed_samples": analysis["failed_samples"],
        "failure_rate": analysis["failure_rate"],
        "failure_counts": analysis["failure_counts"],
        "failure_rates_by_type": failure_rates,
        "prioritized_failures": [
            {
                "failure_type": failure_type,
                "failure_rate": rate,
            }
            for failure_type, rate in prioritized_failures
        ],
        "failed_questions": analysis["failed_questions"],
    }