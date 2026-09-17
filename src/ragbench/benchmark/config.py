from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration for a single RAGBench experiment.

    An experiment defines the RAG parameters that we want to test
    and compare against other configurations.
    """

    name: str

    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 3

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "qwen2.5:1.5b"

    def __post_init__(self):
        """
        Validate the experiment configuration after initialization.
        """

        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty.")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty.")

        if not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty.")