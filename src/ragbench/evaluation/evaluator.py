from typing import Any, Dict, List
from src.ragbench.evaluation.diagnostics import (diagnose_result,)
import time

from src.ragbench.evaluation.optimization_engine import (
    optimize_result,
)

from src.ragbench.evaluation.recommendations import (
    generate_recommendations,
)
from src.ragbench.evaluation.passage_matching import (
    find_relevant_chunks,
    reference_recall_at_k,
)
from src.eval.answer_correctness import answer_correctness_score

from src.ragbench.core.interfaces import RAGSystem
from src.ragbench.datasets.schema import EvaluationSample

from src.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
)

from src.eval.faithfulness import faithfulness_score
from src.eval.answer_relevancy import answer_relevancy_score
from src.eval.answer_correctness import answer_correctness_score


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

        # ----------------------------------------------------
        # Run the RAG system and measure latency
        # ----------------------------------------------------

        start_time = time.perf_counter()

        result = self.rag_system.answer(
            sample.question,
            top_k=top_k,
        )

        latency = time.perf_counter() - start_time

        retrieved_chunks = result["retrieved_chunks"]

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        generated_answer = result["generated_answer"]

        # ----------------------------------------------------
        # Get relevant ground-truth chunks
        # ----------------------------------------------------

        relevant_chunk_ids = find_relevant_chunks(
            retrieved_chunks,
            sample.reference_passages,
        )

        # ----------------------------------------------------
        # Retrieval evaluation
        # ----------------------------------------------------

        recall = reference_recall_at_k(
        retrieved_chunks,
        sample.reference_passages,
        top_k,
)

        precision = precision_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            top_k,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_chunk_ids,
        )

        # ----------------------------------------------------
        # Generation evaluation
        # ----------------------------------------------------

        # Combine retrieved chunks into one context string
        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        # Faithfulness
        faithfulness_result = faithfulness_score(
            generated_answer,
            context,
        )

        # Answer relevancy
        relevancy = answer_relevancy_score(
            sample.question,
            generated_answer,
        )

        # Answer correctness
        correctness = answer_correctness_score(
            generated_answer,
            sample.ground_truth_answer,
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        evaluation_result = {
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
                "answer_correctness": correctness,
            },
            "latency_seconds": latency,
        }

        evaluation_result["diagnostics"] = diagnose_result(
            evaluation_result
        )
        evaluation_result["recommendations"] = (
            generate_recommendations(evaluation_result)
        )

        return evaluation_result

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