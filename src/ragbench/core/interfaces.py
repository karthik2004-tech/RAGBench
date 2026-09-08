from abc import ABC, abstractmethod
from typing import List, Dict, Any


class RAGSystem(ABC):
    """
    Base interface that every RAG system evaluated by RAGBench
    should implement.
    """

    @abstractmethod
    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a question.

        Each result should contain at least:

        {
            "chunk_id": "...",
            "text": "...",
            "score": 0.95
        }
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate an answer using the question and
        retrieved context.
        """
        raise NotImplementedError

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Complete RAG operation.

        This method connects retrieval and generation.
        """

        retrieved_chunks = self.retrieve(
            question,
            top_k=top_k,
        )

        answer = self.generate(
            question,
            retrieved_chunks,
        )

        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": answer,
        }