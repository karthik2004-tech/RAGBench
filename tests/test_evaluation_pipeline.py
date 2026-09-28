from src.ragbench.evaluation.diagnostics import (
    diagnose_result,
)

from src.ragbench.evaluation.recommendations import (
    generate_recommendations,
)


def test_diagnostics_and_recommendations_work_together():

    result = {
        "id": "q001",
        "question": "What is RAG?",

        "retrieval": {
            "recall_at_k": 0.2,
            "precision_at_k": 0.3,
            "reciprocal_rank": 0.2,
        },

        "generation": {
            "faithfulness": {
                "score": 0.3,
                "claims": [],
            },
            "answer_relevancy": 0.8,
            "answer_correctness": 0.4,
        },

        "latency_seconds": 1.2,
    }

    diagnostics = diagnose_result(
        result
    )

    recommendations = generate_recommendations(
        result
    )

    assert diagnostics[
        "overall_status"
    ] == "fail"

    assert (
        "retrieval_failure"
        in diagnostics["failure_types"]
    )

    assert (
        "faithfulness_failure"
        in diagnostics["failure_types"]
    )

    assert isinstance(
        recommendations,
        list,
    )

    assert len(recommendations) > 0