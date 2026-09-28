from src.ragbench.evaluation.failure_analysis import (
    analyze_failures,
)


def make_result(
    sample_id,
    question,
    status,
    primary_diagnosis="no_failure",
    failure_types=None,
):
    if failure_types is None:
        failure_types = []

    return {
        "id": sample_id,
        "question": question,
        "diagnostics": {
            "overall_status": status,
            "primary_diagnosis": primary_diagnosis,
            "failure_types": failure_types,
        },
    }


def test_all_passed():

    results = [
        make_result(
            "q001",
            "What is RAG?",
            "pass",
        ),
        make_result(
            "q002",
            "What are RAG components?",
            "pass",
        ),
    ]

    analysis = analyze_failures(results)

    assert analysis["total_samples"] == 2
    assert analysis["passed_samples"] == 2
    assert analysis["failed_samples"] == 0
    assert analysis["failure_rate"] == 0.0
    assert analysis["failure_counts"] == {}


def test_failure_count():

    results = [
        make_result(
            "q001",
            "Question 1",
            "pass",
        ),
        make_result(
            "q002",
            "Question 2",
            "fail",
            "retrieval_failure",
            ["retrieval_failure"],
        ),
        make_result(
            "q003",
            "Question 3",
            "fail",
            "retrieval_failure",
            ["retrieval_failure"],
        ),
    ]

    analysis = analyze_failures(results)

    assert analysis["total_samples"] == 3
    assert analysis["passed_samples"] == 1
    assert analysis["failed_samples"] == 2

    assert analysis["failure_rate"] == 2 / 3

    assert (
        analysis["failure_counts"]["retrieval_failure"]
        == 2
    )


def test_multiple_failure_types():

    results = [
        make_result(
            "q001",
            "Question 1",
            "fail",
            "retrieval_failure",
            [
                "retrieval_failure",
                "answer_correctness_failure",
            ],
        ),
    ]

    analysis = analyze_failures(results)

    assert (
        analysis["failure_counts"]["retrieval_failure"]
        == 1
    )

    assert (
        analysis["failure_counts"][
            "answer_correctness_failure"
        ]
        == 1
    )


def test_failed_questions_are_recorded():

    results = [
        make_result(
            "q001",
            "What is RAG?",
            "pass",
        ),
        make_result(
            "q002",
            "What is FAISS?",
            "fail",
            "retrieval_failure",
            ["retrieval_failure"],
        ),
    ]

    analysis = analyze_failures(results)

    assert len(analysis["failed_questions"]) == 1

    failed_question = analysis["failed_questions"][0]

    assert failed_question["id"] == "q002"

    assert (
        failed_question["question"]
        == "What is FAISS?"
    )

    assert (
        failed_question["primary_diagnosis"]
        == "retrieval_failure"
    )


def test_empty_results():

    analysis = analyze_failures([])

    assert analysis["total_samples"] == 0
    assert analysis["passed_samples"] == 0
    assert analysis["failed_samples"] == 0
    assert analysis["failure_rate"] == 0.0
    assert analysis["failure_counts"] == {}
    assert analysis["failed_questions"] == []


def test_failure_rates_by_type():

    results = [
        make_result(
            "q001",
            "Question 1",
            "fail",
            "retrieval_failure",
            ["retrieval_failure"],
        ),
        make_result(
            "q002",
            "Question 2",
            "fail",
            "faithfulness_failure",
            ["faithfulness_failure"],
        ),
        make_result(
            "q003",
            "Question 3",
            "fail",
            "faithfulness_failure",
            ["faithfulness_failure"],
        ),
        make_result(
            "q004",
            "Question 4",
            "pass",
        ),
    ]

    analysis = analyze_failures(results)

    assert analysis["failure_rates_by_type"]["retrieval_failure"] == 0.25
    assert analysis["failure_rates_by_type"]["faithfulness_failure"] == 0.5


def test_ranking_failure_is_counted():

    results = [
        make_result(
            "q001",
            "Question 1",
            "fail",
            "ranking_failure",
            ["ranking_failure"],
        ),
        make_result(
            "q002",
            "Question 2",
            "pass",
        ),
    ]

    analysis = analyze_failures(results)

    assert (
        analysis["failure_counts"]["ranking_failure"]
        == 1
    )

    assert (
        analysis["failure_rates_by_type"]["ranking_failure"]
        == 0.5
    )