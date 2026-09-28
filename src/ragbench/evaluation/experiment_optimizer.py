from typing import Any, Dict, List, Optional


METRIC_KEYS = {
    "recall": "recall_at_k",
    "precision": "precision_at_k",
    "reciprocal_rank": "reciprocal_rank",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "answer_correctness": "answer_correctness",
}


def _extract_faithfulness(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    if isinstance(value, dict):
        value = value.get("score")

    if value is None:
        return None

    return float(value)


def _get_metric(
    result: Dict[str, Any],
    metric: str,
) -> Optional[float]:
    """
    Read a metric from either:

    1. A summarized experiment:
       result["metrics"]["recall_at_k"]

    2. A raw evaluation result:
       result["retrieval"]["recall_at_k"]
       result["generation"]["faithfulness"]
    """

    metric_key = METRIC_KEYS.get(
        metric,
        metric,
    )

    # --------------------------------------------------
    # Summary-style result
    # --------------------------------------------------

    metrics = result.get(
        "metrics",
        {},
    )

    if metric_key in metrics:

        value = metrics.get(
            metric_key
        )

        if metric == "faithfulness":
            return _extract_faithfulness(
                value
            )

        if value is None:
            return None

        return float(value)

    # --------------------------------------------------
    # Raw evaluator result
    # --------------------------------------------------

    retrieval = result.get(
        "retrieval",
        {},
    )

    generation = result.get(
        "generation",
        {},
    )

    if metric_key in retrieval:

        value = retrieval.get(
            metric_key
        )

        if value is None:
            return None

        return float(value)

    if metric_key in generation:

        value = generation.get(
            metric_key
        )

        if metric == "faithfulness":
            return _extract_faithfulness(
                value
            )

        if value is None:
            return None

        return float(value)

    return None


def find_best_experiment(
    experiments: List[Dict[str, Any]],
    metric: str = "recall",
) -> Optional[Dict[str, Any]]:

    valid_experiments = []

    for experiment in experiments:

        value = _get_metric(
            experiment,
            metric,
        )

        if value is not None:

            valid_experiments.append(
                (
                    value,
                    experiment,
                )
            )

    if not valid_experiments:
        return None

    _, best_experiment = max(
        valid_experiments,
        key=lambda item: item[0],
    )

    return best_experiment


def compare_with_best_experiment(
    current_result: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    metric: str = "recall",
) -> Dict[str, Any]:

    current_value = _get_metric(
        current_result,
        metric,
    )

    best_experiment = (
        find_best_experiment(
            experiments,
            metric=metric,
        )
    )

    if best_experiment is None:

        return {
            "comparison_available": False,
            "message": (
                "No historical experiment contains "
                f"a valid {metric} value."
            ),
        }

    best_value = _get_metric(
        best_experiment,
        metric,
    )

    improvement = None

    if (
        current_value is not None
        and best_value is not None
    ):

        improvement = round(
            best_value - current_value,
            6,
        )

    return {
        "comparison_available": True,
        "metric": metric,
        "current_value": current_value,
        "best_historical_value": best_value,
        "improvement": improvement,
        "best_experiment": best_experiment,
    }


def generate_experiment_recommendation(
    current_result: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    metric: str = "recall",
) -> Optional[Dict[str, Any]]:

    comparison = (
        compare_with_best_experiment(
            current_result,
            experiments,
            metric=metric,
        )
    )

    if not comparison[
        "comparison_available"
    ]:
        return None

    current_value = comparison[
        "current_value"
    ]

    best_value = comparison[
        "best_historical_value"
    ]

    best_experiment = comparison[
        "best_experiment"
    ]

    if current_value is None:

        return {
            "type": "historical_experiment",
            "metric": metric,
            "recommendation": (
                "Use the historical benchmark "
                "results to select a configuration "
                "for further testing."
            ),
            "evidence": (
                f"The best historical {metric} "
                f"value is {best_value:.3f}."
            ),
            "experiment": best_experiment,
        }

    if best_value <= current_value:

        return {
            "type": "historical_experiment",
            "metric": metric,
            "recommendation": (
                "The current configuration is already "
                "at least as good as the best historical "
                f"experiment for {metric}."
            ),
            "evidence": (
                f"Current {metric} = {current_value:.3f}; "
                f"best historical {metric} = "
                f"{best_value:.3f}."
            ),
            "experiment": best_experiment,
        }

    return {
        "type": "historical_experiment",
        "metric": metric,
        "recommendation": (
            "Consider testing the historical "
            "configuration that achieved the "
            f"highest {metric}."
        ),
        "evidence": (
            f"Current {metric} = {current_value:.3f}; "
            f"best historical {metric} = "
            f"{best_value:.3f}. "
            f"Potential improvement = "
            f"{best_value - current_value:.3f}."
        ),
        "experiment": best_experiment,
    }


def recommend_historical_configuration(
    current_result: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    metric: str = "recall",
) -> Optional[Dict[str, Any]]:
    """
    Recommend a historical configuration that achieved
    a higher value for the requested metric than the
    current result.
    """

    comparison = (
        compare_with_best_experiment(
            current_result,
            experiments,
            metric=metric,
        )
    )

    if not comparison[
        "comparison_available"
    ]:
        return None

    current_value = comparison[
        "current_value"
    ]

    best_value = comparison[
        "best_historical_value"
    ]

    best_experiment = comparison[
        "best_experiment"
    ]

    if current_value is None:

        return {
            "available": True,
            "metric": metric,
            "reason": (
                "No current metric value was available. "
                "A historical configuration is available "
                "for comparison."
            ),
            "recommended_experiment": (
                best_experiment
            ),
        }

    if best_value <= current_value:

        return {
            "available": False,
            "metric": metric,
            "reason": (
                "No historical experiment achieved "
                f"a higher {metric} value than the "
                "current result."
            ),
            "recommended_experiment": None,
        }

    return {
        "available": True,
        "metric": metric,
        "current_value": current_value,
        "historical_value": best_value,
        "improvement": round(
            best_value - current_value,
            6,
        ),
        "reason": (
            f"The historical experiment "
            f"'{best_experiment['name']}' achieved "
            f"{best_value:.3f} for {metric}, compared "
            f"with the current value of "
            f"{current_value:.3f}."
        ),
        "recommended_experiment": (
            best_experiment
        ),
    }

FAILURE_METRICS = {
    "retrieval_failure": "recall",
    "ranking_failure": "reciprocal_rank",
    "faithfulness_failure": "faithfulness",
    "answer_relevancy_failure": "answer_relevancy",
    "answer_correctness_failure": "answer_correctness",
}


def recommend_next_experiment(
    current_result: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    failure_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Recommend the next experiment based on the primary failure type.

    The failure type determines which metric should be optimized.
    Historical experiments are then searched for a configuration
    that achieved a better value for that metric.
    """

    target_metric = FAILURE_METRICS.get(failure_type)

    if target_metric is None:
        return None

    historical_recommendation = recommend_historical_configuration(
        current_result,
        experiments,
        metric=target_metric,
    )

    if historical_recommendation is None:
        return None

    if not historical_recommendation.get("available", False):
        return None

    experiment = historical_recommendation.get("recommended_experiment")

    if experiment is None:
        return None

    return {
        "failure_type": failure_type,
        "target_metric": target_metric,
        "recommendation": (
            f"Run the historical experiment '{experiment['name']}' "
            f"to investigate and improve {target_metric}."
        ),
        "reason": historical_recommendation["reason"],
        "experiment": experiment,
    }