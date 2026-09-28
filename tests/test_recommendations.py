from src.ragbench.evaluation.recommendations import (
    RecommendationConfig,
    generate_recommendations,
)


def make_result(
    recall=1.0,
    precision=1.0,
    reciprocal_rank=1.0,
    faithfulness=1.0,
    relevancy=1.0,
    correctness=1.0,
    latency=0.5,
):
    return {
        "retrieval": {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": reciprocal_rank,
        },
        "generation": {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "answer_correctness": correctness,
        },
        "latency_seconds": latency,
    }


def test_no_recommendation_when_metrics_are_good():

    result = make_result()

    recommendations = generate_recommendations(result)

    assert len(recommendations) == 1

    assert recommendations[0]["area"] == "overall"


def test_low_recall_recommendation():

    result = make_result(
        recall=0.2,
    )

    recommendations = generate_recommendations(result)

    areas = [
        recommendation["area"]
        for recommendation in recommendations
    ]

    assert "retrieval" in areas


def test_low_precision_recommendation():

    result = make_result(
        precision=0.2,
    )

    recommendations = generate_recommendations(result)

    assert any(
        recommendation["area"] == "retrieval"
        for recommendation in recommendations
    )


def test_low_reciprocal_rank_recommendation():

    result = make_result(
        reciprocal_rank=0.2,
    )

    recommendations = generate_recommendations(result)

    assert any(
        recommendation["area"] == "ranking"
        for recommendation in recommendations
    )


def test_low_faithfulness_recommendation():

    result = make_result(
        faithfulness=0.2,
    )

    recommendations = generate_recommendations(result)

    assert any(
        recommendation["area"] == "generation"
        and recommendation["priority"] == "high"
        for recommendation in recommendations
    )


def test_low_relevancy_recommendation():

    result = make_result(
        relevancy=0.2,
    )

    recommendations = generate_recommendations(result)

    assert any(
        recommendation["area"] == "generation"
        for recommendation in recommendations
    )


def test_low_correctness_recommendation():

    result = make_result(
        correctness=0.2,
    )

    recommendations = generate_recommendations(result)

    assert any(
        recommendation["area"] == "generation"
        for recommendation in recommendations
    )


def test_latency_recommendation():

    result = make_result(
        latency=3.0,
    )

    config = RecommendationConfig(
        high_latency_threshold_seconds=2.0,
    )

    recommendations = generate_recommendations(
        result,
        config=config,
    )

    assert any(
        recommendation["area"] == "performance"
        for recommendation in recommendations
    )


def test_multiple_recommendations():

    result = make_result(
        recall=0.2,
        precision=0.2,
        faithfulness=0.2,
        correctness=0.2,
    )

    recommendations = generate_recommendations(result)

    assert len(recommendations) >= 4

def test_ranking_failure_recommendation():
    result = make_result(
        recall=1.0,
        reciprocal_rank=0.333,
    )

    recommendations = generate_recommendations(result)

    ranking_recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation["area"] == "ranking"
    ]

    assert len(ranking_recommendations) == 1
    assert ranking_recommendations[0]["priority"] == "high"
    assert "ranking" in ranking_recommendations[0]["recommendation"].lower()