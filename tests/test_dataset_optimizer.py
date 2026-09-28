from src.ragbench.evaluation.dataset_optimizer import (
    optimize_dataset,
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


def test_dataset_optimization():

    results = [
        make_result(
            "q001",
            "Question 1",
            "fail",
            "faithfulness_failure",
            ["faithfulness_failure"],
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
            "retrieval_failure",
            ["retrieval_failure"],
        ),
        make_result(
            "q004",
            "Question 4",
            "pass",
        ),
    ]

    report = optimize_dataset(results)

    assert report["total_samples"] == 4
    assert report["passed_samples"] == 1
    assert report["failed_samples"] == 3

    assert report["failure_rate"] == 0.75

    assert (
        report["failure_counts"]["faithfulness_failure"]
        == 2
    )

    assert (
        report["failure_counts"]["retrieval_failure"]
        == 1
    )

    assert (
        report["prioritized_failures"][0]["failure_type"]
        == "faithfulness_failure"
    )

    assert (
        report["prioritized_failures"][0]["failure_rate"]
        == 0.5
    )


def test_empty_dataset():

    report = optimize_dataset([])

    assert report["total_samples"] == 0
    assert report["passed_samples"] == 0
    assert report["failed_samples"] == 0
    assert report["failure_rate"] == 0.0
    assert report["failure_counts"] == {}
    assert report["prioritized_failures"] == []
    assert report["failed_questions"] == []