from src.ragbench.evaluation.passage_matching import (
    normalize_text,
    passage_match_score,
    find_relevant_chunks,
    reference_recall_at_k,
)


def test_normalize_text():
    text = "Radar   sensors, detect nearby objects!"

    result = normalize_text(text)

    assert result == "radar sensors detect nearby objects"


def test_exact_passage_match():
    chunk = (
        "Radar sensors detect the distance and speed "
        "of nearby objects."
    )

    reference = (
        "Radar sensors detect the distance and speed "
        "of nearby objects."
    )

    score = passage_match_score(
        chunk,
        reference,
    )

    assert score == 1.0


def test_unrelated_passage_has_low_score():
    chunk = (
        "LiDAR uses laser pulses to build a precise "
        "3D map of the environment."
    )

    reference = (
        "Radar sensors detect the distance and speed "
        "of nearby objects."
    )

    score = passage_match_score(
        chunk,
        reference,
    )

    assert score == 0.0


def test_find_relevant_chunks():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": (
                "Radar sensors detect the distance and speed "
                "of nearby objects."
            ),
        },
        {
            "chunk_id": "chunk_2",
            "text": (
                "LiDAR uses laser pulses to build a precise "
                "3D map of the environment."
            ),
        },
    ]

    reference_passages = [
        (
            "Radar sensors detect the distance and speed "
            "of nearby objects."
        )
    ]

    relevant = find_relevant_chunks(
        retrieved_chunks,
        reference_passages,
    )

    assert relevant == ["chunk_1"]


def test_multi_sentence_reference():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": (
                "Radar sensors detect the distance and speed "
                "of nearby objects."
            ),
        },
        {
            "chunk_id": "chunk_2",
            "text": (
                "Radar sensors work reliably in poor weather."
            ),
        },
    ]

    reference_passages = [
        (
            "Radar sensors detect the distance and speed "
            "of nearby objects. "
            "Radar sensors work reliably in poor weather."
        )
    ]

    relevant = find_relevant_chunks(
        retrieved_chunks,
        reference_passages,
    )

    assert relevant == ["chunk_1", "chunk_2"]


def test_partial_passage_score():
    chunk = (
        "Radar sensors detect the distance and speed "
        "of nearby objects."
    )

    reference = (
        "Radar sensors detect the distance and speed "
        "of nearby objects. "
        "They work reliably in poor weather."
    )

    score = passage_match_score(
        chunk,
        reference,
    )

    assert score == 0.5


def test_irrelevant_chunk_not_returned():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": (
                "LiDAR uses laser pulses to build a precise "
                "3D map of the environment."
            ),
        },
        {
            "chunk_id": "chunk_2",
            "text": (
                "Ultrasonic sensors are used for short-range "
                "detection."
            ),
        },
    ]

    reference_passages = [
        (
            "Radar sensors detect the distance and speed "
            "of nearby objects."
        )
    ]

    relevant = find_relevant_chunks(
        retrieved_chunks,
        reference_passages,
    )

    assert relevant == []


# ---------------------------------------------------------
# Reference Evidence Recall Tests
# ---------------------------------------------------------


def test_reference_recall_full():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Radar sensors detect nearby objects.",
        },
        {
            "chunk_id": "chunk_2",
            "text": "LiDAR builds a precise 3D map.",
        },
    ]

    reference_passages = [
        "Radar sensors detect nearby objects.",
        "LiDAR builds a precise 3D map.",
    ]

    recall = reference_recall_at_k(
        retrieved_chunks,
        reference_passages,
        k=2,
    )

    assert recall == 1.0


def test_reference_recall_partial():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Radar sensors detect nearby objects.",
        },
    ]

    reference_passages = [
        "Radar sensors detect nearby objects.",
        "LiDAR builds a precise 3D map.",
    ]

    recall = reference_recall_at_k(
        retrieved_chunks,
        reference_passages,
        k=1,
    )

    assert recall == 0.5


def test_reference_recall_zero():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Cameras recognize traffic signs.",
        },
    ]

    reference_passages = [
        "Radar sensors detect nearby objects.",
        "LiDAR builds a precise 3D map.",
    ]

    recall = reference_recall_at_k(
        retrieved_chunks,
        reference_passages,
        k=1,
    )

    assert recall == 0.0


def test_reference_recall_respects_k():
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Radar sensors detect nearby objects.",
        },
        {
            "chunk_id": "chunk_2",
            "text": "LiDAR builds a precise 3D map.",
        },
    ]

    reference_passages = [
        "Radar sensors detect nearby objects.",
        "LiDAR builds a precise 3D map.",
    ]

    recall = reference_recall_at_k(
        retrieved_chunks,
        reference_passages,
        k=1,
    )

    assert recall == 0.5


    