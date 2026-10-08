# RAGBench

RAGBench evaluates retrieval-augmented generation systems through retrieval, answer quality, faithfulness, and latency measurements. It also diagnoses failures and compares experiment configurations. It is designed to work with existing RAG pipelines through a small adapter contract.

## Install

Python 3.10 or newer is required. Install the package and command line tool with:

```bash
python -m pip install -e .
```

Install optional dependencies for the repository's original FAISS and Ollama RAG example with `python -m pip install -e ".[v1]"`. A Streamlit dashboard extra is available as `.[dashboard]`. Evaluation models are loaded locally by the existing evaluators; the first evaluation may download model files.

## Complete developer workflow

This is the end-to-end process for testing an existing RAG system. The developer keeps their retriever, generator, vector store, and model; RAGBench calls the system through an adapter and evaluates its outputs.

### 1. Install into the Python environment you will use

Run these commands from the RAGBench project directory. If using a virtual environment, activate it first and use that environment's Python for both installation and later commands:

```powershell
python -m pip install -e .
python -c "from ragbench import RAGBench; print(RAGBench)"
```

### 2. Connect your RAG pipeline

Create a Python class implementing `retrieve` and `generate`. Place it in an importable module, for example `my_app.py` in your working directory:

```python
from ragbench import RAGSystem


class MyRAG(RAGSystem):
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def retrieve(self, question: str, top_k: int = 3):
        # Adapt your retriever's result to this shape.
        results = self.retriever.search(question, top_k=top_k)
        return [
            {
                "chunk_id": item.id,
                "text": item.text,
                "score": item.score,
            }
            for item in results
        ]

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        # Send the question and retrieved context to your generator.
        return self.generator.answer(question, retrieved_chunks)
```

The example constructor accepts dependencies from the developer's app. For the CLI, expose a no-argument `RAGSystem` class or a module level factory configured for your system. Alternatively, wrap ordinary functions with `CallableRAGAdapter`, or use the LangChain and LlamaIndex adapters.

### 3. Create a representative evaluation dataset

For each question, record the expected answer and the passages that support it. Save these records as a JSON list, for example `data/eval_set.json`:

```json
[
  {
    "id": "q1",
    "question": "What does the retriever do?",
    "ground_truth_answer": "It finds passages relevant to the question.",
    "reference_passages": [
      "The retriever searches the document collection for passages related to a user question."
    ]
  }
]
```

Include the question types and edge cases your users care about. The reference passages should contain the evidence needed to answer each question.

### 4. Run the evaluation

From Python, load the dataset and create `EvaluationSample` objects:

```python
import json
from ragbench import RAGBench
from ragbench.datasets import EvaluationSample
from my_app import MyRAG, make_retriever, make_generator

with open("data/eval_set.json", encoding="utf-8") as file:
    dataset = [EvaluationSample(**row) for row in json.load(file)]

bench = RAGBench(MyRAG(make_retriever(), make_generator()))
results = bench.evaluate(dataset, top_k=3)
```

Or use the CLI. Its `--system` option constructs a no-argument class using `module:Class` notation:

```powershell
ragbench evaluate --system my_app:ConfiguredRAG --dataset data/eval_set.json --config ragbench.yaml
```

By default, the CLI writes evaluation results to `<output.directory>/evaluation.json`. You can override that location with `--output`.

### 5. Inspect quality and diagnose failures

Each evaluated question includes retrieval and generation metrics, the retrieved chunks, generated answer, latency, and diagnostic fields. Inspect a result in Python:

```python
diagnosis = bench.diagnose(results[0])
print(diagnosis["primary_diagnosis"])
print(diagnosis["explanations"])
```

The CLI can diagnose either one result or every result in the evaluation JSON file:

```powershell
ragbench diagnose --input data/results/evaluation.json --config ragbench.yaml --output data/results/diagnosis.json
```

Look at the metrics independently. Low recall suggests missing evidence; lower reciprocal rank with adequate recall suggests relevant evidence was retrieved too low in the ranking. Faithfulness includes claim-level support details, while relevancy and correctness are separate answer quality signals.

### 6. Run controlled configuration experiments

If your application can create its RAG system from a configuration, write a factory that accepts `ExperimentConfig` and returns a `RAGSystem`:

```python
def build_rag(config):
    return MyRAG(
        make_retriever(chunk_size=config.chunk_size, top_k=config.top_k),
        make_generator(),
    )
```

Run multiple settings over the same dataset:

```python
from ragbench.benchmark import ExperimentConfig

experiments = bench.benchmark(
    rag_factory=build_rag,
    dataset=dataset,
    configurations=[
        ExperimentConfig(name="topk_1", top_k=1),
        ExperimentConfig(name="topk_3", top_k=3),
        ExperimentConfig(name="larger_chunks", chunk_size=800, chunk_overlap=50),
    ],
)
for experiment in experiments:
    bench.save_experiment(experiment)
```

The CLI accepts the same kind of factory plus a JSON list of experiment configurations:

```powershell
ragbench benchmark --factory my_app:build_rag --dataset data/eval_set.json --configs experiments.json --config ragbench.yaml
```

Results are stored under the configured output directory. You can load and compare them with `bench.load_experiments()` and `bench.compare()`, or create a Markdown comparison with `bench.benchmark_report()`. Each metric is compared separately; one configuration may improve recall while another improves latency or precision.

### 7. Use experiment history to plan the next run

Convert stored benchmark results into optimizer summaries, then ask for recommendations for a result:

```python
from ragbench.benchmark.summary import summarize_experiments

historical = summarize_experiments(bench.load_experiments())
optimization = bench.optimize(results[0], historical)
print(optimization["diagnostics"])
print(optimization["historical_recommendations"])
print(optimization["next_experiment"])
```

The CLI optimizer accepts an evaluation result and a JSON file of these summaries:

```powershell
ragbench optimize --input data/results/evaluation.json --experiments data/results/summaries.json --output data/results/optimization.json
```

To produce a report with diagnoses, faithfulness claims, recommendations, historical comparisons, and the suggested next experiment:

```powershell
ragbench report --input data/results/evaluation.json --experiments data/results/experiments.json --optimization data/results/optimization.json --output data/results/report.md
```

### 8. Repeat and compare

Run the recommended configuration against the same dataset, save it, and compare its metrics with prior runs. This makes the workflow repeatable and shows whether the targeted metric improved, and what tradeoffs occurred in other metrics. RAGBench reports these tradeoffs instead of naming one overall best configuration.

## Test a RAG system running on another computer

RAGBench can evaluate a system running on a different laptop or server if that system exposes an HTTP API that the computer running RAGBench can reach. RAGBench itself runs locally and calls the remote endpoint through a small `RAGSystem` adapter; it does not start or host the remote service.

### 1. Expose an answer endpoint

For full retrieval and faithfulness evaluation, the remote service should accept a question and `top_k`, then return both the retrieved chunks and the generated answer:

```json
{
  "retrieved_chunks": [
    {
      "chunk_id": "doc-1",
      "text": "The retrieved passage text...",
      "score": 0.91
    }
  ],
  "generated_answer": "The answer generated from the retrieved passages."
}
```

The service can use any web framework. Its request should contain `question` and `top_k`. Configure it to listen on a network interface reachable from the RAGBench computer; an address bound only to `localhost` is reachable only on that same machine. Both computers need to be on a network or VPN that permits the connection, and the service port must be allowed through the host firewall.

### 2. Create a remote adapter

The adapter sends the evaluation request and maps the JSON response to the RAGBench contract. This example uses Python's standard library, so it adds no HTTP client dependency:

```python
import json
from urllib.request import Request, urlopen

from ragbench import RAGBench, RAGSystem


class RemoteRAGSystem(RAGSystem):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def _answer(self, question: str, top_k: int) -> dict:
        payload = json.dumps({
            "question": question,
            "top_k": top_k,
        }).encode("utf-8")

        request = Request(
            f"{self.endpoint}/answer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    # The evaluator calls our answer() override, which makes one request.
    # These methods satisfy RAGSystem's interface for other callers.
    def retrieve(self, question: str, top_k: int = 3):
        raise NotImplementedError("Remote retrieval is handled by answer().")

    def generate(self, question: str, retrieved_chunks):
        raise NotImplementedError("Remote generation is handled by answer().")

    def answer(self, question: str, top_k: int = 3) -> dict:
        result = self._answer(question, top_k)
        return {
            "question": question,
            "retrieved_chunks": result["retrieved_chunks"],
            "generated_answer": result["generated_answer"],
        }


bench = RAGBench(RemoteRAGSystem("http://192.168.1.25:8000"))
results = bench.evaluate(dataset, top_k=3)
```

Replace the example address with the remote machine's hostname or IP and the port used by its service. If the endpoint requires authentication, add the appropriate authorization header in `_answer`; keep credentials outside source code, such as in environment variables. Use HTTPS when traffic needs encryption.

RAGBench measures the full request duration, including network time. If the remote service returns only a generated answer and not retrieved chunks, retrieval metrics and evidence-based faithfulness evaluation will not have the information they need. The evaluation dataset can remain on the RAGBench computer, while each question is sent to the remote service for evaluation.

## Connect a RAG system

Implement `RAGSystem` and return chunks with `chunk_id` and `text` fields. A numeric `score` is recommended:

```python
from ragbench import RAGSystem


class MyRAG(RAGSystem):
    def retrieve(self, question: str, top_k: int = 3):
        return [
            {"chunk_id": "doc-1", "text": "Evidence text", "score": 0.91}
        ][:top_k]

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        return "Answer based on the retrieved evidence."
```

For existing functions, `CallableRAGAdapter` supplies the interface. `LangChainAdapter` and `LlamaIndexAdapter` are duck-typed integrations: they do not add either framework as a dependency. They translate LangChain documents and LlamaIndex source nodes into the RAGBench chunk shape. Custom framework versions can also implement `RAGSystem` directly.

## Evaluate

The evaluation dataset is a JSON list of records with `id`, `question`, `ground_truth_answer`, and `reference_passages`:

```json
[
  {
    "id": "q1",
    "question": "What does the system retrieve?",
    "ground_truth_answer": "It retrieves relevant passages.",
    "reference_passages": ["The retriever searches for relevant passages."]
  }
]
```

Use the public API from Python:

```python
from ragbench import RAGBench

bench = RAGBench(MyRAG())
results = bench.evaluate(dataset, top_k=3)
diagnosis = bench.diagnose(results[0])
optimization = bench.optimize(results[0], historical_experiment_summaries)
markdown = bench.report(results)
```

`RAGBench` also accepts an optional `RAGBenchConfig`. Defaults are top-K 3, diagnostic thresholds of 0.5, and `data/results` as the experiment output directory. Load a YAML configuration with `RAGBenchConfig.from_yaml("ragbench.yaml")`:

```yaml
evaluation:
  top_k: 3
thresholds:
  recall: 0.5
  reciprocal_rank: 0.5
  faithfulness: 0.5
  relevancy: 0.5
  correctness: 0.5
  latency_seconds: 2.0
output:
  directory: data/results
```

## Command line

Create a starter configuration, then call your Python RAG class using `module:Class` notation. Dataset paths point to the JSON schema above.

```bash
ragbench init
ragbench evaluate --system my_app:MyRAG --dataset data/eval_set.json --config ragbench.yaml
ragbench diagnose --input data/results/evaluation.json --config ragbench.yaml
ragbench report --input data/results/evaluation.json --output data/results/report.md
```

To run multiple configurations, pass a factory using `module:function` and a JSON list of `ExperimentConfig` fields:

```bash
ragbench benchmark --factory my_app:build_rag --dataset data/eval_set.json --configs experiments.json --config ragbench.yaml
```

`ragbench optimize --input result.json --experiments summaries.json` uses optimizer-ready historical summaries. The Python API can create these using `summarize_experiments` from `ragbench.benchmark.summary`.

## Benchmark, save, compare, and reproduce

```python
from ragbench import RAGBench
from ragbench.benchmark import ExperimentConfig

bench = RAGBench(config=config)
experiments = bench.benchmark(
    rag_factory=build_rag,
    dataset=dataset,
    configurations=[
        ExperimentConfig(name="topk_1", top_k=1),
        ExperimentConfig(name="topk_3", top_k=3),
    ],
)
for experiment in experiments:
    bench.save_experiment(experiment)

comparison = bench.compare()  # Per-metric comparison of saved experiments
saved = bench.load_experiment("topk_1")
rerun = bench.reproduce("topk_1", build_rag, dataset)
report = bench.benchmark_report()
```

Experiment results are JSON files under the configured output directory. Experiment names may contain letters, numbers, dots, underscores, and hyphens. Saving the same name replaces its result. Reproduction reruns the stored configuration against the dataset and factory supplied by the caller.

## Understand results

- **Recall@K** measures how much reference evidence was retrieved.
- **Precision@K** measures how many retrieved chunks matched reference evidence.
- **Reciprocal rank** measures the position of the first relevant chunk.
- **Faithfulness** uses claim-level evidence and NLI signals to estimate whether answer claims are supported by retrieved context. It is an evaluator signal, not proof of factual correctness.
- **Answer relevancy** measures semantic relation to the question; it does not establish correctness.
- **Answer correctness** compares the answer with the supplied ground truth.
- **Latency** measures the RAG answer call, excluding metric model time.

The evaluation result retains claim support decisions, reasons, and evidence. Markdown reports include available claim details. Interpret each metric independently: different configurations can trade recall, precision, answer quality, and latency, so reports do not identify one overall best configuration.

## Tests

Install the development extra and run the suite:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Some tests load local transformer models and can take several minutes on CPU.

## Package layout and project status

The stable entry point is `import ragbench`. The source-tree import `src.ragbench` remains available for existing repository scripts. The public API, optional configuration, adapters, CLI, Markdown reporting, experiment persistence, packaging metadata, and developer documentation are implemented. The framework delegates metric and optimizer logic to the existing evaluation modules.
