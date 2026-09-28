"""
Faithfulness scoring: does the generated answer stick to what the retrieved
context actually supports?

Method:
  1. Decompose the answer into individual claims.
  2. Split retrieved context into evidence sentences.
  3. For each claim, compare it against every evidence sentence using NLI.
  4. Use the strongest entailment score as the claim's evidence score.
  5. Faithfulness score = fraction of claims that are entailed.

This runs entirely on CPU with a small cross-encoder model.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

from transformers import pipeline


_nli_pipeline = None
_embedding_model = None

def _get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _embedding_model

def semantic_similarity_score(
    premise: str,
    hypothesis: str,
) -> float:
    """Return cosine similarity between evidence and claim."""

    model = _get_embedding_model()

    embeddings = model.encode(
        [premise, hypothesis]
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

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
    """Split a generated answer into individual claims."""
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_into_evidence_sentences(context: str) -> list[str]:
    """Split retrieved context into individual evidence sentences."""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        context.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def nli_entailment_score(
    premise: str,
    hypothesis: str,
) -> float:
    """Return the entailment probability for one premise/hypothesis pair."""

    nli = _get_nli_pipeline()

    result = nli(
        {
            "text": premise,
            "text_pair": hypothesis,
        }
    )

    scores = {
        item["label"].lower(): item["score"]
        for item in result
    }

    return float(
        scores.get("entailment", 0.0)
    )


def claim_is_entailed(
    claim: str,
    context: str,
    threshold: float = 0.5,
    semantic_threshold: float = 0.7,
) -> tuple[bool, float]:
    """
    Check whether a claim is supported by the retrieved context.

    Evidence is considered supportive when either:
      1. NLI detects entailment, or
      2. Semantic similarity is sufficiently high.
    """

    evidence_sentences = split_into_evidence_sentences(
        context
    )

    if not evidence_sentences:
        return False, 0.0

    best_nli_score = 0.0
    best_semantic_score = 0.0

    for evidence in evidence_sentences:

        nli_score = nli_entailment_score(
            evidence,
            claim,
        )

        semantic_score = semantic_similarity_score(
            evidence,
            claim,
        )

        best_nli_score = max(
            best_nli_score,
            nli_score,
        )

        best_semantic_score = max(
            best_semantic_score,
            semantic_score,
        )

    supported = (
        best_nli_score >= threshold
        or best_semantic_score >= semantic_threshold
    )

    evidence_score = max(
        best_nli_score,
        best_semantic_score,
    )

    return (
        supported,
        evidence_score,
    )

def faithfulness_score(
    answer: str,
    context: str,
    threshold: float = 0.5,
) -> dict:
    """Return overall faithfulness plus per-claim evidence."""

    claims = split_into_claims(answer)

    if not claims:
        return {
            "score": 0.0,
            "claims": [],
        }

    claim_results = []
    entailed_count = 0

    for claim in claims:

        is_entailed, score = claim_is_entailed(
            claim,
            context,
            threshold,
        )

        claim_results.append(
            {
                "claim": claim,
                "entailed": is_entailed,
                "entailment_score": round(
                    score,
                    3,
                ),
            }
        )

        if is_entailed:
            entailed_count += 1

    return {
        "score": round(
            entailed_count / len(claims),
            3,
        ),
        "claims": claim_results,
    }


if __name__ == "__main__":

    context = (
        "RAG combines a retriever with a language model "
        "that generates grounded answers."
    )

    answer = (
        "RAG combines retrieval and generation. "
        "It was invented in 2024 by OpenAI."
    )

    result = faithfulness_score(
        answer,
        context,
    )

    print(
        f"Faithfulness: {result['score']}"
    )

    for claim in result["claims"]:

        status = (
            "OK"
            if claim["entailed"]
            else "UNSUPPORTED"
        )

        print(
            f"  [{status}] "
            f"{claim['claim']} "
            f"({claim['entailment_score']})"
        )

        semantic_score = semantic_similarity_score(
        context,
        answer,
    )

    print(
        f"Semantic similarity: "
        f"{semantic_score:.3f}"
    )
