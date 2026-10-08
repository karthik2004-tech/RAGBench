"""Command line interface backed by the public RAGBench API."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

from src.ragbench import RAGBench
from src.ragbench.benchmark import ExperimentConfig
from src.ragbench.config import RAGBenchConfig, write_default_config


def _read_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_object(specification: str) -> Any:
    if ":" not in specification:
        raise ValueError("Python entry point must use module:object syntax.")
    module_name, object_name = specification.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for part in object_name.split("."):
        value = getattr(value, part)
    return value


def _samples(path: str) -> List[Any]:
    from src.ragbench.datasets import EvaluationSample

    values = _read_json(path)
    if not isinstance(values, list):
        raise ValueError("Evaluation dataset JSON must contain a list of samples.")
    return [EvaluationSample(**value) for value in values]


def _config(path: str | None) -> RAGBenchConfig:
    return RAGBenchConfig.from_yaml(path) if path else RAGBenchConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragbench", description="Evaluate and compare RAG systems.")
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init", help="Create a starter ragbench.yaml.")
    init_parser.add_argument("--path", default="ragbench.yaml")

    evaluate = commands.add_parser("evaluate", help="Evaluate a RAGSystem over a JSON dataset.")
    evaluate.add_argument("--system", required=True, help="RAGSystem class as module:Class")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--output")

    benchmark = commands.add_parser("benchmark", help="Run configured experiment variants.")
    benchmark.add_argument("--factory", required=True, help="RAG factory as module:function")
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--configs", required=True, help="JSON list of ExperimentConfig mappings")
    benchmark.add_argument("--config")

    diagnose = commands.add_parser("diagnose", help="Diagnose one evaluation result JSON file.")
    diagnose.add_argument("--input", required=True)
    diagnose.add_argument("--config")
    diagnose.add_argument("--output")

    optimize = commands.add_parser("optimize", help="Optimize using historical experiment summaries.")
    optimize.add_argument("--input", required=True)
    optimize.add_argument("--experiments", required=True, help="JSON list of historical experiment summaries")
    optimize.add_argument("--output")

    report = commands.add_parser("report", help="Render evaluation results or experiments as Markdown.")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--optimization", help="Optional optimization report JSON")
    report.add_argument("--experiments", help="Optional historical benchmark results JSON")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        write_default_config(args.path)
        print(f"Created {args.path}")
        return 0

    if args.command == "evaluate":
        system_type = _load_object(args.system)
        bench = RAGBench(system_type(), config=_config(args.config))
        results = bench.evaluate(_samples(args.dataset))
        output = args.output or str(Path(bench.config.output_dir) / "evaluation.json")
        _write_json(output, results)
        print(f"Saved {len(results)} evaluation results to {output}")
        return 0

    if args.command == "benchmark":
        config = _config(args.config)
        factory = _load_object(args.factory)
        experiment_configs = [ExperimentConfig(**row) for row in _read_json(args.configs)]
        bench = RAGBench(config=config)
        runs = bench.benchmark(factory, _samples(args.dataset), experiment_configs)
        for run in runs:
            bench.save_experiment(run)
        print(f"Saved {len(runs)} experiments to {config.output_dir}")
        return 0

    if args.command == "diagnose":
        bench = RAGBench(config=_config(args.config))
        payload = _read_json(args.input)
        report = (
            [bench.diagnose(row) for row in payload]
            if isinstance(payload, list)
            else bench.diagnose(payload)
        )
        rendered = json.dumps(report, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0

    if args.command == "optimize":
        bench = RAGBench()
        payload = _read_json(args.input)
        historical = _read_json(args.experiments)
        result = (
            [bench.optimize(row, historical) for row in payload]
            if isinstance(payload, list)
            else bench.optimize(payload, historical)
        )
        rendered = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0

    if args.command == "report":
        payload = _read_json(args.input)
        bench = RAGBench()
        if isinstance(payload, dict) and "experiment" in payload:
            rendered = bench.benchmark_report([payload])
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict) and "experiment" in payload[0]:
            rendered = bench.benchmark_report(payload)
        else:
            rendered = bench.report(
                payload if isinstance(payload, list) else [payload],
                optimization=_read_json(args.optimization) if args.optimization else None,
                experiments=_read_json(args.experiments) if args.experiments else None,
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Saved report to {args.output}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
