"""Optional YAML configuration for common RAGBench defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from src.ragbench.evaluation.diagnostics import DiagnosticConfig


@dataclass
class RAGBenchConfig:
    top_k: int = 3
    output_dir: str = "data/results"
    thresholds: DiagnosticConfig = field(default_factory=DiagnosticConfig)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        for name in (
            "recall_threshold", "reciprocal_rank_threshold", "faithfulness_threshold",
            "relevancy_threshold", "correctness_threshold",
        ):
            value = float(getattr(self.thresholds, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        latency = self.thresholds.latency_threshold_seconds
        if latency is not None and latency <= 0:
            raise ValueError("latency_threshold_seconds must be greater than 0.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGBenchConfig":
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configuration requires PyYAML.") from error

        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a YAML mapping.")

        evaluation = data.get("evaluation", {})
        thresholds = data.get("thresholds", {})
        output = data.get("output", {})
        if not all(isinstance(section, dict) for section in (evaluation, thresholds, output)):
            raise ValueError("evaluation, thresholds, and output must be mappings.")

        top_k = int(evaluation.get("top_k", 3))
        if top_k <= 0:
            raise ValueError("evaluation.top_k must be greater than 0.")

        names = {
            "recall": "recall_threshold",
            "reciprocal_rank": "reciprocal_rank_threshold",
            "faithfulness": "faithfulness_threshold",
            "relevancy": "relevancy_threshold",
            "correctness": "correctness_threshold",
            "latency_seconds": "latency_threshold_seconds",
        }
        diagnostic_values: Dict[str, Any] = {
            target: thresholds[source]
            for source, target in names.items()
            if source in thresholds
        }
        for name, value in diagnostic_values.items():
            if name.endswith("_threshold") and not 0 <= float(value) <= 1:
                raise ValueError(f"thresholds.{name} must be between 0 and 1.")
            if name == "latency_threshold_seconds" and float(value) <= 0:
                raise ValueError("thresholds.latency_seconds must be greater than 0.")

        return cls(
            top_k=top_k,
            output_dir=str(output.get("directory", "data/results")),
            thresholds=DiagnosticConfig(**diagnostic_values),
        )


def write_default_config(path: str | Path = "ragbench.yaml") -> Path:
    """Write a starter config without replacing an existing file."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Configuration already exists: {destination}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Writing YAML configuration requires PyYAML.") from error
    document = {
        "evaluation": {"top_k": 3},
        "thresholds": {
            "recall": 0.5,
            "reciprocal_rank": 0.5,
            "faithfulness": 0.5,
            "relevancy": 0.5,
            "correctness": 0.5,
        },
        "output": {"directory": "data/results"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return destination
