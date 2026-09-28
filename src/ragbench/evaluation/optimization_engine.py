from typing import Any, Dict, List

from src.ragbench.evaluation.diagnostics import (
    diagnose_result,
)

from src.ragbench.evaluation.recommendations import (
    generate_recommendations,
)

from src.ragbench.evaluation.experiment_optimizer import (
    recommend_historical_configuration,
    recommend_next_experiment
)


def optimize_result(
    current_result: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine diagnostics, recommendations, and
    historical experiment evidence into one
    optimization report.
    """

    diagnostics = diagnose_result(
        current_result
    )

    recommendations = generate_recommendations(
        current_result
    )

    historical_recommendations = []

    metrics = [
        "recall",
        "precision",
        "reciprocal_rank",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    for metric in metrics:

        recommendation = (
            recommend_historical_configuration(
                current_result,
                experiments,
                metric=metric,
            )
        )

        if recommendation is not None:
            historical_recommendations.append(
                recommendation
            )

        next_experiment = recommend_next_experiment(
            current_result,
            experiments,
            failure_type=diagnostics["primary_diagnosis"],
        )

    return {
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "historical_recommendations": 
            historical_recommendations,
        "next_experiment": next_experiment,
        
    }