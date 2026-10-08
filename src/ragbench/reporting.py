"""Human-readable Markdown reports for evaluation and experiment results."""

from typing import Any, Dict, Iterable, List

from src.ragbench.benchmark.comparison import ExperimentComparator
from src.ragbench.benchmark.summary import calculate_average_metrics


def _faithfulness_value(generation: Dict[str, Any]) -> float:
    value = generation.get("faithfulness", 0.0)
    return float(value.get("score", 0.0) if isinstance(value, dict) else value)


def render_report(
    results: Iterable[Dict[str, Any]],
    title: str = "RAGBench Evaluation Report",
    optimization: Dict[str, Any] | None = None,
    experiments: Iterable[Dict[str, Any]] | None = None,
) -> str:
    rows = list(results)
    averages = calculate_average_metrics(rows)
    lines = [f"# {title}", "", f"Samples evaluated: **{len(rows)}**", "", "## Aggregate metrics", "",
             "| Metric | Mean |", "|---|---:|"]
    for key, label in (
        ("recall_at_k", "Recall@K"), ("precision_at_k", "Precision@K"),
        ("reciprocal_rank", "Reciprocal rank"), ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer relevancy"), ("answer_correctness", "Answer correctness"),
        ("latency_seconds", "Latency (seconds)"),
    ):
        lines.append(f"| {label} | {averages[key]:.3f} |")

    lines.extend(["", "## Per question", ""])
    for result in rows:
        retrieval = result.get("retrieval", {})
        generation = result.get("generation", {})
        diagnostic = result.get("diagnostics", {})
        lines.extend([
            f"### {result.get('id', 'Question')}: {result.get('question', '')}", "",
            f"- Diagnosis: {diagnostic.get('primary_diagnosis', 'not available')}",
            f"- Recall@K: {float(retrieval.get('recall_at_k', 0)):.3f}; precision@K: {float(retrieval.get('precision_at_k', 0)):.3f}; reciprocal rank: {float(retrieval.get('reciprocal_rank', 0)):.3f}",
            f"- Faithfulness: {_faithfulness_value(generation):.3f}; relevancy: {float(generation.get('answer_relevancy', 0)):.3f}; correctness: {float(generation.get('answer_correctness', 0)):.3f}",
            f"- Latency: {float(result.get('latency_seconds', 0)):.3f}s", "",
            "**Generated answer**", "", str(result.get("generated_answer", "")), "",
        ])
        for explanation in diagnostic.get("explanations", []):
            lines.append(f"- Failure detail: {explanation}")
        for recommendation in result.get("recommendations", []):
            if isinstance(recommendation, dict):
                description = recommendation.get("recommendation", recommendation.get("reason", str(recommendation)))
            else:
                description = str(recommendation)
            lines.append(f"- Recommendation: {description}")
        lines.append("")
        faithfulness = generation.get("faithfulness", {})
        claims = faithfulness.get("claims", []) if isinstance(faithfulness, dict) else []
        if claims:
            lines.extend(["**Faithfulness claims**", ""])
            for claim in claims:
                lines.append(f"- {'Supported' if claim.get('supported', claim.get('entailed', False)) else 'Unsupported'}: {claim.get('claim', '')}")
                evidence = claim.get("evidence")
                if evidence:
                    lines.append(f"  Evidence: {evidence}")
            lines.append("")

    history = list(experiments or [])
    if history:
        comparison = ExperimentComparator().compare(history)
        lines.extend(["## Historical comparison", "", "Metric leaders are listed independently; this is not an overall ranking.", ""])
        for metric, experiment_name in comparison.get("best", {}).items():
            summary = next(
                (row for row in comparison.get("experiments", []) if row.get("name") == experiment_name),
                {},
            )
            value_key = "avg_latency_seconds" if metric == "latency_seconds" else metric
            value = summary.get(value_key)
            if metric == "latency_seconds":
                lines.append(f"- Lowest latency: **{experiment_name}** ({value:.3f}s)")
            else:
                lines.append(f"- Highest {metric.replace('_', ' ')}: **{experiment_name}** ({value:.3f})")
        lines.append("")

    if optimization:
        next_experiment = optimization.get("next_experiment")
        lines.extend(["## Optimization", ""])
        if next_experiment:
            target = next_experiment.get("target_metric", "the failing metric")
            experiment_name = next_experiment.get("experiment", {}).get("name", "unnamed experiment")
            lines.append(f"- Suggested next experiment: **{experiment_name}** for {target}.")
            reason = next_experiment.get("reason")
            if reason:
                lines.append(f"- Reason: {reason}")
        else:
            lines.append("No next experiment was recommended from the supplied history.")
        for item in optimization.get("historical_recommendations", []):
            if item.get("available"):
                lines.append(f"- Historical improvement for {item.get('metric')}: {item.get('reason')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_benchmark_report(experiments: Iterable[Dict[str, Any]], title: str = "RAGBench Benchmark Report") -> str:
    rows = list(experiments)
    lines = [f"# {title}", "", "Metrics are reported independently because configurations can trade quality and latency.", ""]
    if not rows:
        return "\n".join(lines + ["No experiments to report."])
    lines.extend(["| Experiment | Recall | Precision | Reciprocal rank | Faithfulness | Relevancy | Correctness | Avg latency (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for experiment in rows:
        config = experiment.get("experiment", {})
        metrics = calculate_average_metrics(experiment.get("results", []))
        lines.append(
            f"| {config.get('name', 'unknown')} | {metrics['recall_at_k']:.3f} | {metrics['precision_at_k']:.3f} | {metrics['reciprocal_rank']:.3f} | {metrics['faithfulness']:.3f} | {metrics['answer_relevancy']:.3f} | {metrics['answer_correctness']:.3f} | {metrics['latency_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"
