from typing import Any, Dict, List

from src.rag.retrieve import Retriever
from src.rag.generate import generate_answer
from src.ragbench.core.interfaces import RAGSystem


class V1RAGAdapter(RAGSystem):
    """
    Adapter that connects the original V1 RAG pipeline
    to the RAGBench V2 RAGSystem interface.
    """

    def __init__(
        self,
        index_dir: str = "data/index",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen2.5:1.5b",
    ):
        self.retriever = Retriever(
            index_dir=index_dir,
            model_name=embedding_model,
        )

        self.llm_model = llm_model

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Use the existing V1 FAISS retriever.
        """

        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Use the existing V1 Ollama/Qwen generator.
        """

        return generate_answer(
            question=question,
            chunks=retrieved_chunks,
            model=self.llm_model,
        )