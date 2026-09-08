"""
Retrieval evaluation metrics.

Supports one or multiple relevant chunk IDs.

All functions expect:

retrieved_ids:
    List of chunk IDs returned by the retriever,
    ranked best-first.

relevant_ids:
    List of chunk IDs considered relevant.
"""


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Recall@K = number of relevant chunks retrieved in top-K
               divided by total number of relevant chunks.
    """

    if not relevant_ids:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    hits = len(retrieved_top_k & relevant)

    return hits / len(relevant)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Precision@K = relevant chunks in top-K / number retrieved in top-K.
    """

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant = set(relevant_ids)

    hits = sum(
        1
        for chunk_id in top_k
        if chunk_id in relevant
    )

    return hits / len(top_k)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Reciprocal Rank = 1 / rank of the first relevant chunk.
    Returns 0 if no relevant chunk is retrieved.
    """

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_retrieved_ids: list[list[str]],
    all_relevant_ids: list[list[str]],
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.
    """

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant
        in zip(all_retrieved_ids, all_relevant_ids)
    ]

    return sum(scores) / len(scores) if scores else 0.0