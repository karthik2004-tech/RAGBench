"""
Faithfulness scoring: does the generated answer stick to what the retrieved
context actually supports?

Method (inspired by RAGAS's faithfulness metric, reimplemented locally):
  1. Decompose the answer into individual sentences/claims.
  2. For each claim, check whether it's entailed by the retrieved context
     using a local NLI (natural language inference) model.
  3. Faithfulness score = fraction of claims that are entailed.

This runs entirely on CPU with a small cross-encoder model — no API calls.
"""
import re

from transformers import pipeline

_nli_pipeline = None


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
    """Naive sentence splitter. Swap for a proper sentence tokenizer
    (e.g. nltk.sent_tokenize) if your answers have complex punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def claim_is_entailed(claim: str, context: str, threshold: float = 0.5) -> tuple[bool, float]:
    """Check whether `context` entails `claim` using the NLI model.

    cross-encoder/nli-deberta-v3-small outputs labels: contradiction, entailment, neutral.
    We treat "entailment" probability above `threshold` as supported.
    """
    nli = _get_nli_pipeline()
    # NLI convention: premise = context, hypothesis = claim
    result = nli(f"{context} </s></s> {claim}")[0]
    scores = {r["label"].lower(): r["score"] for r in result}
    entail_score = scores.get("entailment", 0.0)
    return entail_score >= threshold, entail_score


def faithfulness_score(answer: str, context: str, threshold: float = 0.5) -> dict:
    """Returns overall faithfulness score plus per-claim breakdown for the dashboard."""
    claims = split_into_claims(answer)
    if not claims:
        return {"score": 0.0, "claims": []}

    claim_results = []
    entailed_count = 0
    for claim in claims:
        is_entailed, score = claim_is_entailed(claim, context, threshold)
        claim_results.append({
            "claim": claim,
            "entailed": is_entailed,
            "entailment_score": round(score, 3),
        })
        if is_entailed:
            entailed_count += 1

    return {
        "score": round(entailed_count / len(claims), 3),
        "claims": claim_results,
    }


if __name__ == "__main__":
    context = "RAG combines a retriever with a language model that generates grounded answers."
    answer = "RAG combines retrieval and generation. It was invented in 2024 by OpenAI."
    result = faithfulness_score(answer, context)
    print(f"Faithfulness: {result['score']}")
    for c in result["claims"]:
        print(f"  [{'OK' if c['entailed'] else 'UNSUPPORTED'}] {c['claim']} ({c['entailment_score']})")
