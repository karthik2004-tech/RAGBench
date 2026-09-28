from src.ragbench.evaluation.diagnostics import (
    DiagnosticConfig,
    diagnose_result,
)


def make_result(
    recall=1.0,
    reciprocal_rank=1.0,
    faithfulness=1.0,
    relevancy=1.0,
    correctness=1.0,
    latency=0.5,
):
    return {
        "retrieval": {
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
        },
        "generation": {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "answer_correctness": correctness,
        },
        "latency_seconds": latency,
    }


def test_successful_result():

    result = make_result()

    diagnosis = diagnose_result(result)

    assert diagnosis["overall_status"] == "pass"
    assert diagnosis["primary_diagnosis"] == "no_failure"
    assert diagnosis["failure_types"] == []


def test_retrieval_failure():

    result = make_result(
        recall=0.2,
        reciprocal_rank=0.2,
    )

    diagnosis = diagnose_result(result)

    assert diagnosis["overall_status"] == "fail"

    assert (
        diagnosis["primary_diagnosis"]
        == "retrieval_failure"
    )

    assert (
        "retrieval_failure"
        in diagnosis["failure_types"]
    )


def test_faithfulness_failure():

    result = make_result(
        faithfulness=0.2,
    )

    diagnosis = diagnose_result(result)

    assert (
        "faithfulness_failure"
        in diagnosis["failure_types"]
    )


def test_answer_relevancy_failure():

    result = make_result(
        relevancy=0.2,
    )

    diagnosis = diagnose_result(result)

    assert (
        "answer_relevancy_failure"
        in diagnosis["failure_types"]
    )


def test_answer_correctness_failure():

    result = make_result(
        correctness=0.2,
    )

    diagnosis = diagnose_result(result)

    assert (
        "answer_correctness_failure"
        in diagnosis["failure_types"]
    )


def test_multiple_failures():

    result = make_result(
        recall=0.2,
        reciprocal_rank=0.2,
        faithfulness=0.2,
        relevancy=0.2,
    )

    diagnosis = diagnose_result(result)

    assert diagnosis["overall_status"] == "fail"

    assert "retrieval_failure" in diagnosis["failure_types"]

    assert "faithfulness_failure" in diagnosis["failure_types"]

    assert "answer_relevancy_failure" in diagnosis["failure_types"]


def test_latency_failure():

    result = make_result(
        latency=3.0,
    )

    config = DiagnosticConfig(
        latency_threshold_seconds=2.0,
    )

    diagnosis = diagnose_result(
        result,
        config=config,
    )

    assert "latency_issue" in diagnosis["failure_types"]


def test_custom_threshold():

    result = make_result(
        recall=0.6,
    )

    config = DiagnosticConfig(
        recall_threshold=0.7,
    )

    diagnosis = diagnose_result(
        result,
        config=config,
    )

    assert "retrieval_failure" in diagnosis["failure_types"]