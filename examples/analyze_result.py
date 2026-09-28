from src.ragbench.evaluation.diagnostics import (
    diagnose_result,
)

from src.ragbench.evaluation.recommendations import (
    generate_recommendations,
)


def main():

    result = {
        "id": "demo",
        "question": "What is RAG?",

        "retrieval": {
            "recall_at_k": 0.4,
            "precision_at_k": 0.3,
            "reciprocal_rank": 0.4,
        },

        "generation": {
            "faithfulness": {
                "score": 0.2,
                "claims": [],
            },
            "answer_relevancy": 0.8,
            "answer_correctness": 0.5,
        },

        "latency_seconds": 1.4,
    }

    diagnostics = diagnose_result(
        result
    )

    recommendations = generate_recommendations(
        result
    )

    print("=" * 60)
    print("RAGBench Diagnosis")
    print("=" * 60)

    print(
        f"\nStatus: "
        f"{diagnostics['overall_status']}"
    )

    print(
        f"Primary diagnosis: "
        f"{diagnostics['primary_diagnosis']}"
    )

    print("\nFailure types:")

    for failure in diagnostics[
        "failure_types"
    ]:
        print(f"  - {failure}")

    print("\nExplanation:")

    for explanation in diagnostics[
        "explanations"
    ]:
        print(f"  - {explanation}")

    print("\nRecommendations:")

    for recommendation in recommendations:

        print(
            f"\n  [{recommendation['priority']}] "
            f"{recommendation['area']}"
        )

        print(
            f"  {recommendation['recommendation']}"
        )

        print(
            f"  Reason: "
            f"{recommendation['reason']}"
        )


if __name__ == "__main__":
    main()