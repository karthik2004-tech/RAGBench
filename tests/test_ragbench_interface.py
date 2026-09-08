from examples.custom_rag.example_rag import ExampleRAG


def test_rag_system_retrieve():

    rag = ExampleRAG()

    results = rag.retrieve(
        "What sensors are used in ADAS?",
        top_k=3,
    )

    assert isinstance(results, list)

    assert len(results) > 0

    assert "chunk_id" in results[0]

    assert "text" in results[0]

    assert "score" in results[0]


def test_rag_system_answer():

    rag = ExampleRAG()

    result = rag.answer(
        "What sensors are used in ADAS?",
        top_k=3,
    )

    assert "question" in result

    assert "retrieved_chunks" in result

    assert "generated_answer" in result

    assert isinstance(
        result["generated_answer"],
        str,
    )