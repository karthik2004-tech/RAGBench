"""Claim-level faithfulness scoring against retrieved evidence."""

import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


_nli_pipeline = None
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def semantic_similarity_score(premise: str, hypothesis: str) -> float:
    """Return cosine similarity between evidence and claim."""
    model = _get_embedding_model()
    embeddings = model.encode([premise, hypothesis])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        _nli_pipeline = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            top_k=None,
        )
    return _nli_pipeline


def split_into_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims."""
    if not answer or not answer.strip():
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]


def split_into_evidence_sentences(context: str) -> list[str]:
    """Split retrieved text into evidence sentences, ignoring blank lines."""
    if not context or not context.strip():
        return []
    # Newlines from chunk wrapping are whitespace inside a sentence; punctuation
    # still defines evidence boundaries. This preserves paragraph-spanning text.
    normalized = re.sub(r"\s+", " ", context.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]


def nli_entailment_score(premise: str, hypothesis: str) -> float:
    """Return the entailment probability for one premise/hypothesis pair."""
    result = _get_nli_pipeline()({"text": premise, "text_pair": hypothesis})
    scores = {item["label"].lower(): item["score"] for item in result}
    return float(scores.get("entailment", 0.0))


def _claim_evidence_result(
    claim: str,
    context: str,
    threshold: float = 0.5,
    semantic_threshold: float = 0.7,
) -> dict:
    evidence_sentences = split_into_evidence_sentences(context)
    if not evidence_sentences:
        return {
            "claim": claim,
            "supported": False,
            "entailed": False,
            "best_evidence": None,
            "nli_score": 0.0,
            "semantic_similarity": 0.0,
            "entailment_score": 0.0,
            "reason": "No retrieved evidence was available to support this claim.",
        }

    evidence_results = []
    for evidence in evidence_sentences:
        nli_score = nli_entailment_score(evidence, claim)
        similarity = semantic_similarity_score(evidence, claim)
        evidence_results.append((evidence, nli_score, similarity))

    # Retain the strongest evidence by the existing score convention while
    # reporting both underlying signals for that same evidence sentence.
    best_evidence, best_nli, best_similarity = max(
        evidence_results, key=lambda item: max(item[1], item[2])
    )
    supported_by_nli = best_nli >= threshold
    supported_by_similarity = best_similarity >= semantic_threshold
    supported = supported_by_nli or supported_by_similarity
    if supported_by_nli and supported_by_similarity:
        reason = "The evidence passes both the NLI entailment and semantic similarity checks."
    elif supported_by_nli:
        reason = "The evidence passes the NLI entailment check."
    elif supported_by_similarity:
        reason = "The evidence passes the semantic similarity check."
    else:
        reason = "No retrieved evidence sufficiently supports this claim."

    score = max(best_nli, best_similarity)
    return {
        "claim": claim,
        "supported": supported,
        "entailed": supported,  # Backwards-compatible field used by existing consumers.
        "best_evidence": best_evidence,
        "nli_score": round(best_nli, 3),
        "semantic_similarity": round(best_similarity, 3),
        "entailment_score": round(score, 3),  # Legacy field; maximum signal, not calibrated NLI.
        "reason": reason,
    }


def claim_is_entailed(
    claim: str,
    context: str,
    threshold: float = 0.5,
    semantic_threshold: float = 0.7,
) -> tuple[bool, float]:
    """Backward-compatible boolean/score helper."""
    result = _claim_evidence_result(claim, context, threshold, semantic_threshold)
    return result["supported"], result["entailment_score"]


def faithfulness_score(
    answer: str,
    context: str,
    threshold: float = 0.5,
    semantic_threshold: float = 0.7,
) -> dict:
    """Return the fraction of claims supported, with diagnostics per claim."""
    claims = split_into_claims(answer)
    if not claims:
        return {"score": 0.0, "claims": []}

    claim_results = [
        _claim_evidence_result(claim, context, threshold, semantic_threshold)
        for claim in claims
    ]
    supported_count = sum(result["supported"] for result in claim_results)
    return {
        "score": round(supported_count / len(claims), 3),
        "claims": claim_results,
    }


if __name__ == "__main__":
    context = "RAG combines a retriever with a language model that generates grounded answers."
    answer = "RAG combines retrieval and generation. It was invented in 2024 by OpenAI."
    result = faithfulness_score(answer, context)
    print(f"Faithfulness: {result['score']:.3f}")
    for index, claim in enumerate(result["claims"], 1):
        print(f"\nClaim {index}: {claim['claim']}")
        print(f"NLI: {claim['nli_score']:.3f}")
        print(f"Semantic similarity: {claim['semantic_similarity']:.3f}")
        print(f"Supported: {'YES' if claim['supported'] else 'NO'}")
        print(f"Reason: {claim['reason']}")
        if claim["best_evidence"]:
            print(f"Evidence: {claim['best_evidence']}")
