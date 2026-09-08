from typing import Any, Dict, List

from src.ragbench.core.interfaces import RAGSystem
from src.ragbench.datasets.schema import EvaluationSample

from src.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
)

from src.eval.faithfulness import faithfulness_score
from src.eval.answer_relevancy import answer_relevancy_score


class RAGEvaluator:
    """
    Main RAGBench evaluation engine.

    It accepts any RAG system implementing the
    RAGSystem interface.
    """

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        top_k: int = 3,
    ) -> Dict[str, Any]:

        # Run the RAG system
        result = self.rag_system.answer(
            sample.question,
            top_k=top_k,
        )

        retrieved_chunks = result["retrieved_chunks"]

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        generated_answer = result["generated_answer"]

        # ----------------------------------------------------
        # Retrieval evaluation
        # ----------------------------------------------------

        # V1 retrieval metrics currently support one ground-truth ID.
        

        recall = recall_at_k(
            retrieved_ids,
            sample.relevant_chunk_ids,
            top_k,
        )

        precision = precision_at_k(
            retrieved_ids,
            sample.relevant_chunk_ids,
            top_k,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            sample.relevant_chunk_ids,
        )

        # ----------------------------------------------------
        # Generation evaluation
        # ----------------------------------------------------

        # Combine retrieved chunks into one context string
        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        faithfulness_result = faithfulness_score(
            generated_answer,
            context,
        )

        relevancy = answer_relevancy_score(
            sample.question,
            generated_answer,
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {
            "id": sample.id,
            "question": sample.question,

            "retrieved_chunks": retrieved_chunks,

            "generated_answer": generated_answer,

            "retrieval": {
                "recall_at_k": recall,
                "precision_at_k": precision,
                "reciprocal_rank": rr,
            },

            "generation": {
                "faithfulness": faithfulness_result,
                "answer_relevancy": relevancy,
            },
        }

    def evaluate(
        self,
        dataset: List[EvaluationSample],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:

        results = []

        for sample in dataset:

            result = self.evaluate_sample(
                sample,
                top_k=top_k,
            )

            results.append(result)

        return results