"""Duck-typed LlamaIndex adapter; LlamaIndex remains optional."""

from typing import Any, Dict, List

from src.ragbench.core.interfaces import RAGSystem


class LlamaIndexAdapter(RAGSystem):
    """Adapt a LlamaIndex query engine and its source nodes to RAGBench."""

    def __init__(self, query_engine: Any):
        self.query_engine = query_engine
        self._cached_question: str | None = None
        self._cached_response: Any = None

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        response = self.query_engine.query(question)
        self._cached_question = question
        self._cached_response = response
        chunks = []
        for index, source in enumerate(list(getattr(response, "source_nodes", []))[:top_k]):
            node = getattr(source, "node", source)
            if hasattr(node, "get_content"):
                text = node.get_content()
            else:
                text = getattr(node, "text", None)
            if not isinstance(text, str):
                raise ValueError("LlamaIndex source nodes must expose text content.")
            metadata = getattr(node, "metadata", {}) or {}
            chunks.append({
                "chunk_id": str(metadata.get("chunk_id", getattr(node, "node_id", index))),
                "text": text,
                "score": float(getattr(source, "score", 0.0) or 0.0),
            })
        return chunks

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        response = self._cached_response
        if self._cached_question != question or response is None:
            response = self.query_engine.query(question)
        return str(getattr(response, "response", response))
