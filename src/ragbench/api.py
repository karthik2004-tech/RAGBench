"""Stable, developer-facing entry point for RAGBench."""

from typing import Any, Callable, Dict, Iterable, List

from src.ragbench.benchmark import ExperimentConfig, ExperimentRunner
from src.ragbench.benchmark import ExperimentComparator, ExperimentStorage
from src.ragbench.config import RAGBenchConfig
from src.ragbench.core.interfaces import RAGSystem
from src.ragbench.datasets.schema import EvaluationSample
from src.ragbench.evaluation.diagnostics import DiagnosticConfig, diagnose_result
from src.ragbench.evaluation.evaluator import RAGEvaluator
from src.ragbench.evaluation.optimization_engine import optimize_result
from src.ragbench.reporting import render_benchmark_report, render_report


class RAGBench:
    """Convenience facade over RAGBench's evaluation and experiment tools.

    A system is optional at construction so a single facade can also be used
    to diagnose or optimize previously saved evaluation results.
    """

    def __init__(self, rag_system: RAGSystem | None = None, config: RAGBenchConfig | None = None):
        self.rag_system = rag_system
        self.config = config or RAGBenchConfig()

    def evaluate(
        self,
        dataset: List[EvaluationSample],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate the configured RAG system against a dataset."""
        if self.rag_system is None:
            raise ValueError("evaluate() requires a RAGSystem passed to RAGBench.")
        return RAGEvaluator(self.rag_system, diagnostic_config=self.config.thresholds).evaluate(
            dataset,
            top_k=self.config.top_k if top_k is None else top_k,
        )

    def diagnose(
        self,
        result: Dict[str, Any],
        config: DiagnosticConfig | None = None,
    ) -> Dict[str, Any]:
        """Diagnose one existing evaluation result."""
        return diagnose_result(result, config=config or self.config.thresholds)

    def optimize(
        self,
        result: Dict[str, Any],
        experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build an optimization report using historical experiments."""
        return optimize_result(result, experiments, config=self.config.thresholds)

    def benchmark(
        self,
        rag_factory: Callable[[ExperimentConfig], RAGSystem],
        dataset: List[EvaluationSample],
        configurations: Iterable[ExperimentConfig],
    ) -> List[Dict[str, Any]]:
        """Run each supplied configuration over the same dataset."""
        runner = ExperimentRunner(rag_factory=rag_factory, dataset=dataset)
        return [runner.run(config) for config in configurations]

    def save_experiment(self, result: Dict[str, Any]) -> str:
        """Persist a complete benchmark result in the configured output directory."""
        return ExperimentStorage(self.config.output_dir).save(result)

    def load_experiment(self, name: str) -> Dict[str, Any]:
        """Load a previously saved experiment."""
        return ExperimentStorage(self.config.output_dir).load(name)

    def load_experiments(self) -> List[Dict[str, Any]]:
        """Load every saved experiment in the configured output directory."""
        return ExperimentStorage(self.config.output_dir).load_all()

    def reproduce(
        self,
        name: str,
        rag_factory: Callable[[ExperimentConfig], RAGSystem],
        dataset: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """Rerun a stored experiment configuration over a supplied dataset."""
        saved = self.load_experiment(name)
        config = ExperimentConfig(**saved["experiment"])
        return ExperimentRunner(rag_factory=rag_factory, dataset=dataset).run(config)

    def compare(self, experiments: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Compare saved or explicitly supplied experiments metric by metric."""
        return ExperimentComparator().compare(experiments if experiments is not None else self.load_experiments())

    def report(
        self,
        results: List[Dict[str, Any]],
        title: str = "RAGBench Evaluation Report",
        optimization: Dict[str, Any] | None = None,
        experiments: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Render evaluation results as Markdown, preserving claim details."""
        return render_report(results, title=title, optimization=optimization, experiments=experiments)

    def benchmark_report(
        self,
        experiments: List[Dict[str, Any]] | None = None,
        title: str = "RAGBench Benchmark Report",
    ) -> str:
        """Render experiments as a metric-by-metric Markdown comparison."""
        return render_benchmark_report(
            experiments if experiments is not None else self.load_experiments(),
            title=title,
        )
