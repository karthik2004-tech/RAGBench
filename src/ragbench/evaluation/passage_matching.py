import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def token_overlap_score(
    chunk_text: str,
    reference_text: str,
) -> float:
    """
    Measure how much of the reference evidence
    is present in the retrieved chunk.

    Score:

        shared reference tokens
        -----------------------
        total reference tokens
    """

    chunk_tokens = set(
        tokenize(chunk_text)
    )

    reference_tokens = set(
        tokenize(reference_text)
    )

    if not reference_tokens:
        return 0.0

    shared_tokens = (
        chunk_tokens
        & reference_tokens
    )

    return (
        len(shared_tokens)
        / len(reference_tokens)
    )


def passage_match_score(
    chunk_text: str,
    reference_passage: str,
) -> float:
    """
    Calculate evidence coverage between a
    retrieved chunk and a reference passage.

    A complete sentence match receives 1.0.

    Partial sentence matches are evaluated using
    reference-token coverage.
    """

    chunk = normalize_text(
        chunk_text
    )

    reference_sentences = (
        split_into_sentences(
            reference_passage
        )
    )

    if not reference_sentences:
        return 0.0

    sentence_scores = []

    for sentence in reference_sentences:

        normalized_sentence = (
            normalize_text(sentence)
        )

        # Exact containment
        if normalized_sentence in chunk:
            sentence_scores.append(1.0)
            continue

        # Partial evidence coverage
        score = token_overlap_score(
            chunk_text,
            sentence,
        )

        sentence_scores.append(
            score
        )

    return max(
        sentence_scores
    )


def find_relevant_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    reference_passages: List[str],
    threshold: float = 0.5,
) -> List[str]:
    """
    Identify retrieved chunks that contain
    sufficient evidence from reference passages.

    A chunk is considered relevant when its
    reference-token coverage reaches the threshold.
    """

    relevant_chunk_ids = []

    reference_sentences = []

    for reference_passage in reference_passages:

        reference_sentences.extend(
            split_into_sentences(
                reference_passage
            )
        )

    for chunk in retrieved_chunks:

        chunk_text = chunk["text"]

        for reference_sentence in (
            reference_sentences
        ):

            score = passage_match_score(
                chunk_text,
                reference_sentence,
            )

            if score >= threshold:

                relevant_chunk_ids.append(
                    chunk["chunk_id"]
                )

                break

    return relevant_chunk_ids


def reference_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    reference_passages: List[str],
    k: int,
    threshold: float = 0.5,
) -> float:
    """
    Measure how much of the reference evidence
    was retrieved in the top-k chunks.

    Recall:

        matched reference sentences
        ----------------------------
        total reference sentences
    """

    top_k_chunks = retrieved_chunks[:k]

    reference_sentences = []

    for reference_passage in reference_passages:

        reference_sentences.extend(
            split_into_sentences(
                reference_passage
            )
        )

    normalized_references = []

    for sentence in reference_sentences:

        normalized = normalize_text(
            sentence
        )

        if (
            normalized
            and normalized
            not in normalized_references
        ):
            normalized_references.append(
                normalized
            )

    if not normalized_references:
        return 0.0

    matched = 0

    for reference_sentence in (
        normalized_references
    ):

        found = False

        for chunk in top_k_chunks:

            score = passage_match_score(
                chunk["text"],
                reference_sentence,
            )

            if score >= threshold:

                found = True
                break

        if found:
            matched += 1

    return (
        matched
        / len(normalized_references)
    )