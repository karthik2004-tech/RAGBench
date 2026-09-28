from src.ragbench.evaluation.passage_matching import (
    normalize_text,
    token_overlap_score,
    passage_match_score,
    find_relevant_chunks,
    reference_recall_at_k,
)


def test_normalize_text():

    text = "Hello,   WORLD!"

    assert normalize_text(text) == (
        "hello world"
    )


def test_exact_passage_match():

    chunk = (
        "Radar sensors detect the distance "
        "and speed of nearby objects."
    )

    reference = (
        "Radar sensors detect the distance "
        "and speed of nearby objects."
    )

    score = passage_match_score(
        chunk,
        reference,
    )

    assert score == 1.0


def test_partial_sentence_match():

    chunk = (
        "a retriever, which finds relevant text "
        "chunks from a knowledge base, with a "
        "language model, which generates an answer "
        "conditioned on those chunks."
    )

    reference = (
        "Retrieval-Augmented Generation combines "
        "a retriever, which finds relevant text "
        "chunks from a knowledge base, with a "
        "language model, which generates an answer "
        "conditioned on those chunks."
    )

    score = passage_match_score(
        chunk,
        reference,
    )

    assert score >= 0.5


def test_find_relevant_chunks_with_partial_match():

    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": (
                "a retriever, which finds relevant "
                "text chunks from a knowledge base, "
                "with a language model."
            ),
        },
        {
            "chunk_id": "chunk_2",
            "text": (
                "Unrelated information about cars."
            ),
        },
    ]

    reference = [
        (
            "Retrieval-Augmented Generation combines "
            "a retriever, which finds relevant text "
            "chunks from a knowledge base, with a "
            "language model."
        )
    ]

    relevant = find_relevant_chunks(
        chunks,
        reference,
    )

    assert "chunk_1" in relevant
    assert "chunk_2" not in relevant


def test_reference_recall_with_partial_match():

    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": (
                "a retriever, which finds relevant "
                "text chunks from a knowledge base, "
                "with a language model."
            ),
        }
    ]

    references = [
        (
            "Retrieval-Augmented Generation combines "
            "a retriever, which finds relevant text "
            "chunks from a knowledge base, with a "
            "language model."
        )
    ]

    recall = reference_recall_at_k(
        chunks,
        references,
        k=1,
    )

    assert recall == 1.0