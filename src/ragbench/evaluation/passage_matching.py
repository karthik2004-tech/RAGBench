import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def passage_match_score(
    chunk_text: str,
    reference_passage: str,
) -> float:
    """
    Calculate how much of the reference passage
    is contained in the retrieved chunk.

    Returns a score between 0 and 1.
    """

    chunk = normalize_text(chunk_text)

    reference_sentences = split_into_sentences(
        reference_passage
    )

    if not reference_sentences:
        return 0.0

    matched_sentences = 0

    for sentence in reference_sentences:
        normalized_sentence = normalize_text(sentence)

        if normalized_sentence in chunk:
            matched_sentences += 1

    return matched_sentences / len(reference_sentences)


def find_relevant_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    reference_passages: List[str],
    threshold: float = 1.0,
) -> List[str]:
    """
    Identify retrieved chunks that contain at least one
    complete sentence from a reference passage.
    """

    relevant_chunk_ids = []

    # Split all reference passages into individual sentences
    reference_sentences = []

    for reference_passage in reference_passages:
        reference_sentences.extend(
            split_into_sentences(reference_passage)
        )

    # Normalize reference sentences
    normalized_references = [
        normalize_text(sentence)
        for sentence in reference_sentences
        if normalize_text(sentence)
    ]

    # Check every retrieved chunk
    for chunk in retrieved_chunks:
        chunk_text = normalize_text(chunk["text"])

        # A chunk is relevant if it contains
        # at least one complete reference sentence
        for reference_sentence in normalized_references:

            if reference_sentence in chunk_text:
                relevant_chunk_ids.append(
                    chunk["chunk_id"]
                )
                break

    return relevant_chunk_ids

def reference_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    reference_passages: List[str],
    k: int,
) -> float:
    """
    Measure how much of the reference evidence was retrieved.

    Recall = matched reference sentences / total reference sentences.
    """

    # Only consider the top-k retrieved chunks
    top_k_chunks = retrieved_chunks[:k]

    # Extract all reference sentences
    reference_sentences = []

    for reference_passage in reference_passages:
        reference_sentences.extend(
            split_into_sentences(reference_passage)
        )

    # Normalize and remove duplicate reference sentences
    normalized_references = []

    for sentence in reference_sentences:
        normalized = normalize_text(sentence)

        if normalized and normalized not in normalized_references:
            normalized_references.append(normalized)

    # No reference evidence
    if not normalized_references:
        return 0.0

    # Count how many reference sentences
    # appear in the retrieved top-k chunks
    matched = 0

    for reference_sentence in normalized_references:
        found = False

        for chunk in top_k_chunks:
            chunk_text = normalize_text(chunk["text"])

            if reference_sentence in chunk_text:
                found = True
                break

        if found:
            matched += 1

    return matched / len(normalized_references)