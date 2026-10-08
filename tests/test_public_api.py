from src.ragbench import RAGBench, RAGSystem, RAGEvaluator
from ragbench import RAGBench as InstalledStyleRAGBench
from src.ragbench.benchmark import ExperimentConfig
from src.ragbench.config import RAGBenchConfig
from src.ragbench.evaluation.diagnostics import DiagnosticConfig


class DummyRAG(RAGSystem):
    def retrieve(self, question, top_k=3):
        return []

    def generate(self, question, retrieved_chunks):
        return "answer"


def test_public_api_exports_preserve_existing_types():
    assert RAGBench
    assert RAGEvaluator
    assert RAGSystem
    assert InstalledStyleRAGBench


def test_evaluate_uses_existing_evaluator():
    bench = RAGBench(DummyRAG())

    assert bench.evaluate([], top_k=2) == []


def test_evaluate_requires_a_configured_rag_system():
    try:
        RAGBench().evaluate([])
    except ValueError as error:
        assert "RAGSystem" in str(error)
    else:
        raise AssertionError("evaluate() should require a RAGSystem")


def test_diagnose_delegates_to_existing_diagnostics():
    result = {
        "retrieval": {"recall_at_k": 1.0, "reciprocal_rank": 1.0},
        "generation": {
            "faithfulness": {"score": 1.0},
            "answer_relevancy": 1.0,
            "answer_correctness": 1.0,
        },
    }

    assert RAGBench().diagnose(result)["overall_status"] == "pass"


def test_optimize_delegates_to_existing_optimization_engine():
    result = {
        "retrieval": {"recall_at_k": 1.0, "precision_at_k": 1.0, "reciprocal_rank": 1.0},
        "generation": {
            "faithfulness": {"score": 1.0, "claims": []},
            "answer_relevancy": 1.0,
            "answer_correctness": 1.0,
        },
    }

    report = RAGBench().optimize(result, [])

    assert report["diagnostics"]["overall_status"] == "pass"
    assert "recommendations" in report
    assert "next_experiment" in report


def test_api_uses_configured_thresholds_for_diagnosis():
    result = {
        "retrieval": {"recall_at_k": 0.8, "reciprocal_rank": 0.8},
        "generation": {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "answer_correctness": 1.0,
        },
    }
    bench = RAGBench(config=RAGBenchConfig(thresholds=DiagnosticConfig(recall_threshold=0.9)))
    assert bench.diagnose(result)["primary_diagnosis"] == "retrieval_failure"


def test_experiment_can_be_saved_loaded_compared_and_reproduced(tmp_path):
    bench = RAGBench(config=RAGBenchConfig(output_dir=str(tmp_path)))
    experiment = {
        "experiment": {
            "name": "saved", "chunk_size": 400, "chunk_overlap": 50,
            "top_k": 3, "embedding_model": "embed", "llm_model": "llm",
        },
        "results": [],
    }
    bench.save_experiment(experiment)
    assert bench.load_experiment("saved") == experiment
    assert bench.compare()["experiments"] == []
    rerun = bench.reproduce("saved", lambda config: DummyRAG(), [])
    assert rerun["experiment"]["name"] == "saved"


def test_benchmark_delegates_to_existing_experiment_runner():
    config = ExperimentConfig(name="public_api", top_k=2)
    results = RAGBench().benchmark(lambda _: DummyRAG(), [], [config])

    assert len(results) == 1
    assert results[0]["experiment"]["name"] == "public_api"
    assert results[0]["results"] == []
