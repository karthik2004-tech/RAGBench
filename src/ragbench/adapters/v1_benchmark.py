import os
from typing import Any, Dict, List

from src.rag.index import build_index
from src.rag.retrieve import Retriever
from src.rag.generate import generate_answer

from src.ragbench.core.interfaces import RAGSystem
from src.ragbench.benchmark.config import ExperimentConfig


class V1BenchmarkRAG(RAGSystem):
    """
    V1 RAG implementation used for RAGBench experiments.

    Unlike the normal V1 adapter, this class builds an experiment-specific
    FAISS index using the chunking parameters from ExperimentConfig.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        corpus_dir: str = "data/corpus",
        experiment_dir: str = "data/experiments",
    ):
        self.config = config

        self.corpus_dir = corpus_dir
        self.experiment_dir = experiment_dir

        # Create a unique index directory for this experiment.
        self.index_dir = os.path.join(
            experiment_dir,
            config.name,
            "index",
        )

        # Build an index using this experiment's parameters.
        build_index(
            corpus_dir=self.corpus_dir,
            out_dir=self.index_dir,
            model_name=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Load the experiment-specific index.
        self.retriever = Retriever(
            index_dir=self.index_dir,
            model_name=config.embedding_model,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:

        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

    def get_relevant_chunk_ids(
        self,
        relevant_documents: List[str],
    ) -> List[str]:
        """
        Return all chunks belonging to the relevant documents
        for this experiment's chunking configuration.
        """

        relevant_documents = set(relevant_documents)

        return [
            record["chunk_id"]
            for record in self.retriever.records
            if record["doc_id"] in relevant_documents
        ]

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:

        return generate_answer(
            question=question,
            chunks=retrieved_chunks,
            model=self.config.llm_model,
        )