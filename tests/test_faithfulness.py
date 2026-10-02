from src.eval.faithfulness import faithfulness_score


def test_supported_claim_is_faithful(monkeypatch):

    class FakeNLI:
        def __call__(self, inputs):
            return [
                {"label": "entailment", "score": 0.95},
                {"label": "neutral", "score": 0.04},
                {"label": "contradiction", "score": 0.01},
            ]

    monkeypatch.setattr(
        "src.eval.faithfulness._get_nli_pipeline",
        lambda: FakeNLI(),
    )
    monkeypatch.setattr(
        "src.eval.faithfulness.semantic_similarity_score",
        lambda premise, hypothesis: 0.91,
    )

    result = faithfulness_score(
        "RAG uses retrieved context to generate an answer.",
        "RAG uses retrieved context to generate an answer.",
    )

    assert result["score"] == 1.0
    assert result["claims"][0]["entailed"] is True


def test_unsupported_claim_is_not_faithful(monkeypatch):

    class FakeNLI:
        def __call__(self, inputs):
            return [
                {"label": "neutral", "score": 0.90},
                {"label": "contradiction", "score": 0.08},
                {"label": "entailment", "score": 0.02},
            ]

    monkeypatch.setattr(
        "src.eval.faithfulness._get_nli_pipeline",
        lambda: FakeNLI(),
    )
    monkeypatch.setattr(
        "src.eval.faithfulness.semantic_similarity_score",
        lambda premise, hypothesis: 0.2,
    )

    result = faithfulness_score(
        "RAG was invented in 2024.",
        "RAG uses retrieved context to generate an answer.",
    )

    assert result["score"] == 0.0
    assert result["claims"][0]["entailed"] is False

def test_semantic_similarity_can_support_claim(monkeypatch):

    from src.eval import faithfulness

    monkeypatch.setattr(
        faithfulness,
        "nli_entailment_score",
        lambda premise, hypothesis: 0.01,
    )

    monkeypatch.setattr(
        faithfulness,
        "semantic_similarity_score",
        lambda premise, hypothesis: 0.8,
    )

    supported, score = faithfulness.claim_is_entailed(
        "Radar detects nearby objects.",
        "Radar sensors detect the distance and speed of nearby objects.",
    )

    assert supported is True
    assert score == 0.8


def test_low_nli_and_low_semantic_similarity_is_not_supported(
    monkeypatch,
):

    from src.eval import faithfulness

    monkeypatch.setattr(
        faithfulness,
        "nli_entailment_score",
        lambda premise, hypothesis: 0.1,
    )

    monkeypatch.setattr(
        faithfulness,
        "semantic_similarity_score",
        lambda premise, hypothesis: 0.3,
    )

    supported, score = faithfulness.claim_is_entailed(
        "The system uses quantum computing.",
        "Radar sensors detect nearby objects.",
    )

    assert supported is False
    assert score == 0.3


def _mock_signals(monkeypatch, *, nli, similarity):
    from src.eval import faithfulness

    monkeypatch.setattr(faithfulness, "nli_entailment_score", lambda p, h: nli(p, h))
    monkeypatch.setattr(
        faithfulness, "semantic_similarity_score", lambda p, h: similarity(p, h)
    )


def test_direct_entailment_has_claim_level_evidence(monkeypatch):
    _mock_signals(monkeypatch, nli=lambda p, h: 0.91, similarity=lambda p, h: 0.82)
    result = faithfulness_score(
        "Radar detects nearby objects.",
        "Radar detects nearby objects.",
    )
    claim = result["claims"][0]
    assert result["score"] == 1.0
    assert claim["supported"] is True
    assert claim["nli_score"] == 0.91
    assert claim["semantic_similarity"] == 0.82
    assert claim["best_evidence"] == "Radar detects nearby objects."


def test_paraphrase_supported_by_existing_semantic_threshold(monkeypatch):
    _mock_signals(monkeypatch, nli=lambda p, h: 0.32, similarity=lambda p, h: 0.74)
    result = faithfulness_score(
        "A retriever and language model form the main RAG components.",
        "RAG combines a retriever with a language model.",
    )
    assert result["score"] == 1.0
    assert result["claims"][0]["supported"] is True
    assert "semantic similarity" in result["claims"][0]["reason"]


def test_unsupported_claim_exposes_reason_and_scores(monkeypatch):
    _mock_signals(monkeypatch, nli=lambda p, h: 0.12, similarity=lambda p, h: 0.41)
    result = faithfulness_score("The system uses quantum computing.", "Radar detects nearby objects.")
    claim = result["claims"][0]
    assert result["score"] == 0.0
    assert claim["supported"] is False
    assert claim["nli_score"] == 0.12
    assert claim["semantic_similarity"] == 0.41
    assert claim["best_evidence"] == "Radar detects nearby objects."
    assert "No retrieved evidence" in claim["reason"]


def test_multiple_claims_aggregate_fraction_supported(monkeypatch):
    _mock_signals(
        monkeypatch,
        nli=lambda p, h: 0.9 if "Radar" in h else 0.1,
        similarity=lambda p, h: 0.8 if "Radar" in h else 0.2,
    )
    result = faithfulness_score(
        "Radar detects nearby objects. The system uses quantum computing.",
        "Radar detects nearby objects.",
    )
    assert result["score"] == 0.5
    assert [claim["supported"] for claim in result["claims"]] == [True, False]


def test_empty_context_marks_claim_unsupported_without_model_calls(monkeypatch):
    _mock_signals(monkeypatch, nli=lambda p, h: 1.0, similarity=lambda p, h: 1.0)
    result = faithfulness_score("Some claim.", "  ")
    assert result["score"] == 0.0
    assert result["claims"][0]["best_evidence"] is None
    assert result["claims"][0]["nli_score"] == 0.0


def test_empty_answer_has_zero_score_and_no_claims():
    assert faithfulness_score("  ", "Evidence exists.") == {"score": 0.0, "claims": []}


def test_chunk_800_q002_regression_uses_actual_answer_and_retrieved_evidence(monkeypatch):
    import json
    from pathlib import Path

    from src.eval import faithfulness

    artifact = json.loads(
        Path("data/experiments/chunk_800/result.json").read_text(encoding="utf-8")
    )
    row = next(item for item in artifact["results"] if item["id"] == "q002")
    answer = row["generated_answer"]
    context = "\n\n".join(chunk["text"] for chunk in row["retrieved_chunks"])
    assert answer == "The main components of a RAG (Retrieval-Augmented Generation) system are a retriever and a language model."

    monkeypatch.setattr(
        faithfulness,
        "nli_entailment_score",
        lambda premise, hypothesis: 0.92 if "Retrieval-Augmented Generation combines a retriever" in premise else 0.05,
    )
    monkeypatch.setattr(
        faithfulness,
        "semantic_similarity_score",
        lambda premise, hypothesis: 0.83 if "Retrieval-Augmented Generation combines a retriever" in premise else 0.2,
    )
    result = faithfulness_score(answer, context)
    claim = result["claims"][0]
    assert result["score"] == 1.0
    assert claim["supported"] is True
    assert "retriever" in claim["best_evidence"]
