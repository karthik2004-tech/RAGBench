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
