"""Adapter for custom retriever and generator callables."""

from typing import Any, Callable, Dict, List

from src.ragbench.core.interfaces import RAGSystem


class CallableRAGAdapter(RAGSystem):
    """Adapt ordinary Python functions without framework dependencies."""

    def __init__(
        self,
        retrieve: Callable[[str, int], List[Dict[str, Any]]],
        generate: Callable[[str, List[Dict[str, Any]]], str],
    ):
        self._retrieve = retrieve
        self._generate = generate

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        chunks = self._retrieve(question, top_k)
        if not isinstance(chunks, list):
            raise TypeError("retrieve callable must return a list of chunk dictionaries.")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError(f"Retrieved chunk {index} must be a dictionary with string 'text'.")
            chunk.setdefault("chunk_id", str(index))
            chunk.setdefault("score", 0.0)
        return chunks

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        answer = self._generate(question, retrieved_chunks)
        if not isinstance(answer, str):
            raise TypeError("generate callable must return a string.")
        return answer
