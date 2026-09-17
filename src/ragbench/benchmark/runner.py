from typing import Callable, List, Dict, Any

from src.ragbench.benchmark.config import ExperimentConfig
from src.ragbench.core.interfaces import RAGSystem
from src.ragbench.datasets.schema import EvaluationSample
from src.ragbench.evaluation.evaluator import RAGEvaluator


class ExperimentRunner:
    """
    Runs a single RAGBench experiment.

    The runner is responsible for:
    1. Creating a RAG system from an experiment configuration.
    2. Running the evaluation dataset.
    3. Returning the evaluation results.
    """

    def __init__(
        self,
        rag_factory: Callable[[ExperimentConfig], RAGSystem],
        dataset: List[EvaluationSample],
    ):
        self.rag_factory = rag_factory
        self.dataset = dataset

    def run(
        self,
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        """
        Run one experiment.

        Parameters
        ----------
        config:
            Configuration describing the experiment.

        Returns
        -------
        Dict containing experiment configuration
        and evaluation results.
        """

        # Create the RAG system using the experiment configuration.
        rag_system = self.rag_factory(config)

        # Create the evaluator for this RAG system.
        evaluator = RAGEvaluator(rag_system)

        # Evaluate the complete dataset.
        results = evaluator.evaluate(
            self.dataset,
            top_k=config.top_k,
        )

        return {
            "experiment": {
                "name": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "top_k": config.top_k,
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
            },
            "results": results,
        }