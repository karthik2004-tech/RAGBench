from src.ragbench.benchmark import (
    ExperimentConfig,
    ExperimentRunner,
)
from src.ragbench.core import RAGSystem


class DummyRAG(RAGSystem):
    """
    Small fake RAG system used to test the experiment runner.

    This avoids loading embeddings, FAISS, Ollama, etc.
    """

    def retrieve(self, question: str, top_k: int = 3):
        return [
            {
                "chunk_id": "doc_1",
                "text": "RAG combines retrieval with generation.",
                "score": 0.9,
            }
        ][:top_k]

    def generate(self, question: str, retrieved_chunks):
        return "RAG combines retrieval with generation."


def dummy_rag_factory(config):
    return DummyRAG()


def test_experiment_runner_creates_results():

    dataset = []

    runner = ExperimentRunner(
        rag_factory=dummy_rag_factory,
        dataset=dataset,
    )

    config = ExperimentConfig(
        name="test_experiment",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )

    result = runner.run(config)

    assert "experiment" in result
    assert "results" in result

    assert result["experiment"]["name"] == "test_experiment"
    assert result["experiment"]["chunk_size"] == 200
    assert result["experiment"]["chunk_overlap"] == 20
    assert result["experiment"]["top_k"] == 3

    assert result["results"] == []


def test_experiment_runner_uses_rag_factory():

    created_configs = []

    def tracking_factory(config):
        created_configs.append(config)
        return DummyRAG()

    runner = ExperimentRunner(
        rag_factory=tracking_factory,
        dataset=[],
    )

    config = ExperimentConfig(
        name="tracking_test",
        chunk_size=600,
        chunk_overlap=50,
        top_k=5,
    )

    runner.run(config)

    assert len(created_configs) == 1
    assert created_configs[0].name == "tracking_test"
    assert created_configs[0].chunk_size == 600
    assert created_configs[0].chunk_overlap == 50
    assert created_configs[0].top_k == 5