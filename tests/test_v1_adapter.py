from src.ragbench.adapters.v1_adapter import V1RAGAdapter


def test_v1_adapter_retrieve():
    rag = V1RAGAdapter()

    results = rag.retrieve(
        "What sensors are commonly used in ADAS?",
        top_k=3,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    assert "chunk_id" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]


def test_v1_adapter_implements_rag_system():
    rag = V1RAGAdapter()

    result = rag.answer(
        "What sensors are commonly used in ADAS?",
        top_k=3,
    )

    assert "question" in result
    assert "retrieved_chunks" in result
    assert "generated_answer" in result

    assert isinstance(result["generated_answer"], str)