"""
Retrieval evaluation metrics, implemented from first principles.

All functions take:
    retrieved_ids: list of chunk_ids returned by the retriever, ranked best-first
    ground_truth_id: the single correct chunk_id for this query

(Extend to multiple ground-truth ids per query if your eval set needs it —
these functions assume one relevant chunk per question, which is the common
case for a hand-curated eval set of this size.)
"""


def recall_at_k(retrieved_ids: list[str], ground_truth_id: str, k: int) -> float:
    """1.0 if the ground-truth chunk appears in the top-k retrieved, else 0.0."""
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def precision_at_k(retrieved_ids: list[str], ground_truth_id: str, k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant.

    With a single ground-truth chunk per query, this is either 1/k (hit) or 0 (miss) —
    included for completeness and to generalize cleanly if you extend the eval set
    to multiple relevant chunks per question.
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid == ground_truth_id)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[str], ground_truth_id: str) -> float:
    """1 / rank of the first correct hit; 0 if not found."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid == ground_truth_id:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(all_retrieved_ids: list[list[str]], all_ground_truth_ids: list[str]) -> float:
    """MRR averaged across a full eval set."""
    scores = [
        reciprocal_rank(retrieved, gt)
        for retrieved, gt in zip(all_retrieved_ids, all_ground_truth_ids)
    ]
    return sum(scores) / len(scores) if scores else 0.0
