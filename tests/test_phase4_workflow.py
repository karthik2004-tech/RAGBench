import json

import pytest

from src.ragbench.adapters import CallableRAGAdapter
from src.ragbench.adapters import LangChainAdapter, LlamaIndexAdapter
from src.ragbench.benchmark import ExperimentConfig
from src.ragbench.config import RAGBenchConfig, write_default_config
from src.ragbench.reporting import render_benchmark_report, render_report


def test_callable_adapter_normalizes_retrieved_chunks():
    adapter = CallableRAGAdapter(
        retrieve=lambda question, top_k: [{"text": "evidence"}],
        generate=lambda question, chunks: "answer",
    )

    chunks = adapter.retrieve("question", top_k=1)
    assert chunks == [{"text": "evidence", "chunk_id": "0", "score": 0.0}]
    assert adapter.generate("question", chunks) == "answer"


def test_callable_adapter_validates_retriever_output():
    adapter = CallableRAGAdapter(lambda question, top_k: ["bad"], lambda question, chunks: "answer")
    with pytest.raises(ValueError, match="chunk"):
        adapter.retrieve("question")


def test_langchain_adapter_uses_duck_typed_components():
    class Document:
        page_content = "retrieved evidence"
        metadata = {"chunk_id": "lc-1", "score": 0.8}

    class Retriever:
        def invoke(self, question):
            return [Document()]

    class Chain:
        def invoke(self, inputs):
            assert inputs["context"] == "retrieved evidence"
            return {"answer": "supported answer"}

    adapter = LangChainAdapter(Retriever(), Chain())
    chunks = adapter.retrieve("question")
    assert chunks[0]["chunk_id"] == "lc-1"
    assert adapter.generate("question", chunks) == "supported answer"


def test_llamaindex_adapter_maps_source_nodes_and_response():
    class Node:
        node_id = "li-1"
        metadata = {}

        def get_content(self):
            return "retrieved evidence"

    class ScoredNode:
        node = Node()
        score = 0.75

    class Response:
        source_nodes = [ScoredNode()]
        response = "generated answer"

    class QueryEngine:
        def query(self, question):
            return Response()

    adapter = LlamaIndexAdapter(QueryEngine())
    chunks = adapter.retrieve("question")
    assert chunks[0]["chunk_id"] == "li-1"
    assert chunks[0]["score"] == 0.75
    assert adapter.generate("question", chunks) == "generated answer"


def test_yaml_config_loads_defaults_and_thresholds(tmp_path):
    path = tmp_path / "ragbench.yaml"
    path.write_text(
        "evaluation:\n  top_k: 5\nthresholds:\n  recall: 0.7\noutput:\n  directory: results\n",
        encoding="utf-8",
    )

    config = RAGBenchConfig.from_yaml(path)
    assert config.top_k == 5
    assert config.thresholds.recall_threshold == 0.7
    assert config.output_dir == "results"


def test_yaml_config_rejects_invalid_top_k(tmp_path):
    path = tmp_path / "ragbench.yaml"
    path.write_text("evaluation:\n  top_k: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top_k"):
        RAGBenchConfig.from_yaml(path)


def test_init_config_does_not_overwrite(tmp_path):
    path = tmp_path / "ragbench.yaml"
    write_default_config(path)
    with pytest.raises(FileExistsError):
        write_default_config(path)


def test_evaluation_report_preserves_claim_details():
    report = render_report([{
        "id": "q1", "question": "Why?", "generated_answer": "Because.",
        "retrieval": {"recall_at_k": 1, "precision_at_k": 1, "reciprocal_rank": 1},
        "generation": {"faithfulness": {"score": 1, "claims": [
            {"claim": "Because.", "supported": True, "evidence": "source text"},
        ]}, "answer_relevancy": 0.9, "answer_correctness": 0.8},
        "latency_seconds": 0.1,
    }])
    assert "Because." in report
    assert "source text" in report
    assert "Faithfulness" in report


def test_evaluation_report_includes_causes_history_and_next_experiment():
    report = render_report(
        [{
            "id": "q2", "question": "Question", "generated_answer": "Answer",
            "retrieval": {"recall_at_k": 0.2, "precision_at_k": 0.2, "reciprocal_rank": 0.2},
            "generation": {"faithfulness": {"score": 0.8}, "answer_relevancy": 0.8, "answer_correctness": 0.8},
            "diagnostics": {"primary_diagnosis": "retrieval_failure", "explanations": ["Not enough evidence."]},
            "recommendations": [{"recommendation": "Increase retrieval coverage."}],
            "latency_seconds": 0.2,
        }],
        optimization={
            "next_experiment": {"target_metric": "recall", "experiment": {"name": "topk_5"}, "reason": "Improved recall."},
            "historical_recommendations": [],
        },
        experiments=[{
            "experiment": {"name": "topk_5", "chunk_size": 400, "chunk_overlap": 50, "top_k": 5},
            "results": [{
                "retrieval": {"recall_at_k": 1.0, "precision_at_k": 0.5, "reciprocal_rank": 1.0},
                "generation": {"faithfulness": {"score": 0.8}, "answer_relevancy": 0.8, "answer_correctness": 0.8},
                "latency_seconds": 0.4,
            }],
        }],
    )
    assert "Not enough evidence." in report
    assert "Increase retrieval coverage." in report
    assert "Historical comparison" in report
    assert "Suggested next experiment: **topk_5**" in report


def test_benchmark_report_keeps_metrics_separate():
    report = render_benchmark_report([{
        "experiment": {"name": "topk_1"},
        "results": [{
            "retrieval": {"recall_at_k": 0.5, "precision_at_k": 1, "reciprocal_rank": 1},
            "generation": {"faithfulness": {"score": 0.6}, "answer_relevancy": 0.7, "answer_correctness": 0.8},
            "latency_seconds": 0.2,
        }],
    }])
    assert "trade" in report
    assert "topk_1" in report
    assert "overall best" not in report.lower()


def test_cli_init_diagnose_and_report(tmp_path):
    from src.ragbench.cli import main

    config = tmp_path / "config.yaml"
    assert main(["init", "--path", str(config)]) == 0
    assert RAGBenchConfig.from_yaml(config).top_k == 3

    result_file = tmp_path / "result.json"
    result = {
        "retrieval": {"recall_at_k": 1, "reciprocal_rank": 1},
        "generation": {"faithfulness": 1, "answer_relevancy": 1, "answer_correctness": 1},
    }
    result_file.write_text(json.dumps([result]), encoding="utf-8")
    diagnosis_file = tmp_path / "diagnosis.json"
    assert main(["diagnose", "--input", str(result_file), "--output", str(diagnosis_file)]) == 0
    assert json.loads(diagnosis_file.read_text(encoding="utf-8"))[0]["overall_status"] == "pass"

    report_file = tmp_path / "report.md"
    assert main(["report", "--input", str(result_file), "--output", str(report_file)]) == 0
    assert "RAGBench Evaluation Report" in report_file.read_text(encoding="utf-8")


def test_cli_loads_evaluation_samples_from_json(tmp_path):
    from src.ragbench.cli import _samples
    from src.ragbench.datasets import EvaluationSample

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps([{
        "id": "q1", "question": "Question", "ground_truth_answer": "Answer",
        "reference_passages": ["Evidence"],
    }]), encoding="utf-8")

    samples = _samples(str(dataset_file))
    assert isinstance(samples[0], EvaluationSample)
    assert samples[0].id == "q1"


def test_experiment_config_is_available_to_benchmark_api():
    assert ExperimentConfig(name="reproducible", top_k=2).top_k == 2


def test_storage_rejects_unsafe_experiment_names(tmp_path):
    from src.ragbench.benchmark import ExperimentStorage

    storage = ExperimentStorage(str(tmp_path))
    with pytest.raises(ValueError, match="Experiment name"):
        storage.save({"experiment": {"name": "../escape"}, "results": []})
