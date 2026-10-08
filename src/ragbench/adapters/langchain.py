"""Duck-typed LangChain adapter; LangChain remains an optional dependency."""

from typing import Any, Dict, List

from src.ragbench.core.interfaces import RAGSystem


class LangChainAdapter(RAGSystem):
    """Connect a LangChain retriever and runnable to RAGBench.

    The retriever must expose ``invoke`` or ``get_relevant_documents``. The
    chain must expose ``invoke`` and accept ``question`` and ``context`` keys.
    """

    def __init__(self, retriever: Any, chain: Any, answer_key: str = "answer"):
        self.retriever = retriever
        self.chain = chain
        self.answer_key = answer_key

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "invoke"):
            documents = self.retriever.invoke(question)
        elif hasattr(self.retriever, "get_relevant_documents"):
            documents = self.retriever.get_relevant_documents(question)
        else:
            documents = self.retriever(question)
        chunks = []
        for index, document in enumerate(list(documents)[:top_k]):
            metadata = getattr(document, "metadata", {}) or {}
            text = getattr(document, "page_content", None)
            if text is None and isinstance(document, dict):
                text = document.get("text", document.get("page_content"))
                metadata = document.get("metadata", metadata)
            if not isinstance(text, str):
                raise ValueError("LangChain documents must expose page_content or text.")
            chunks.append({
                "chunk_id": str(metadata.get("chunk_id", metadata.get("id", index))),
                "text": text,
                "score": float(metadata.get("score", 0.0)),
            })
        return chunks

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
        result = self.chain.invoke({"question": question, "context": context})
        if isinstance(result, dict):
            result = result.get(self.answer_key, result.get("output", result.get("text")))
        if not isinstance(result, str):
            raise TypeError("LangChain chain must return a string or an answer mapping.")
        return result
