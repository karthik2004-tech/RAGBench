# RAGBench — RAG Evaluation & Optimization Framework

RAGBench is an open-source framework for evaluating, benchmarking, and eventually optimizing **Retrieval-Augmented Generation (RAG)** systems.

The project started as a single RAG pipeline with built-in evaluation and is evolving into a reusable, framework-agnostic evaluation platform that can be used with different RAG implementations.

---

## 🚀 Project Goal

A RAG system generating an answer does not necessarily mean that the answer is good.

RAGBench focuses on answering:

> **How well does a RAG system retrieve information, generate grounded answers, and where does it fail?**

The long-term goal is to provide developers with a practical tool to:

- Evaluate RAG quality
- Compare different RAG configurations
- Identify retrieval and generation failures
- Measure answer reliability
- Diagnose RAG failures
- Optimize RAG pipelines based on measurable results

---

# 🏗️ Architecture

## V1 — Baseline RAG Pipeline

```text
Documents
    ↓
Chunking
    ↓
Sentence Transformers
    ↓
Embeddings
    ↓
FAISS Vector Search
    ↓
Top-K Retrieval
    ↓
Retrieved Context
    ↓
Qwen 2.5 1.5B
    ↓
Generated Answer
    ↓
Evaluation
```

---

## V2 — RAGBench Evaluation Framework

The V2 architecture separates the evaluation framework from the original RAG implementation.

```text
                         Any RAG System
                              ↓
                         RAGSystem API
                              ↓
                         RAGEvaluator
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
       Retrieval Evaluation            Generation Evaluation
              ↓                               ↓
    Recall / Precision / RR        Faithfulness / Relevancy
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                    Benchmark & Comparison
                              ↓
                       Evaluation Results
                              ↓
                    Failure Diagnosis
                              ↓
                    RAG Optimization
```

The framework is designed so that an external RAG system can be evaluated without requiring it to use the same retriever, embedding model, vector search engine, or LLM as RAGBench.

---

# ✨ Features

## 🔎 Retrieval Evaluation

RAGBench currently supports:

- Recall@K
- Precision@K
- Reciprocal Rank
- Mean Reciprocal Rank (MRR)
- Multiple reference passages
- Sentence-level reference evidence matching
- Evidence-based Recall@K

### Evidence-based Recall

Instead of only checking whether a retrieved chunk is considered relevant, RAGBench measures how much of the required reference evidence was actually retrieved.

```text
Reference Passages
        ↓
Reference Sentences
        ↓
Retrieved Top-K Chunks
        ↓
Matched Evidence
        ↓
Evidence Coverage
        ↓
Recall@K
```

This makes Recall more meaningful for questions where multiple pieces of evidence are required.

---

## 🤖 Generation Evaluation

RAGBench supports:

- Faithfulness evaluation using Natural Language Inference (NLI)
- Answer Relevancy using semantic similarity
- Answer Correctness
- Per-claim faithfulness analysis

> **Note:** Faithfulness is currently implemented using a local NLI-based heuristic. It should therefore be interpreted as an evaluation signal rather than an absolute measure of factual correctness.

---

## 🧩 Framework-Agnostic RAG Interface

RAG systems can implement the standard RAGBench interface:

```python
class RAGSystem:
    retrieve(...)
    generate(...)
    answer(...)
```

This creates a common interface between external RAG systems and RAGBench.

A developer can therefore evaluate a custom RAG implementation without rewriting the evaluation framework.

---

## 🔌 Adapters

Existing RAG pipelines can be connected to RAGBench through adapters.

The project currently includes an adapter for the original V1 RAG pipeline.

```text
Existing RAG
     ↓
Adapter
     ↓
RAGSystem Interface
     ↓
RAGBench Evaluator
     ↓
Metrics
```

---

## 🧪 Automated Testing

RAGBench includes automated tests covering:

- RAG system interface
- Retrieval metrics
- Passage matching
- Multiple reference evidence
- Evidence-based Recall
- V1 adapter
- Benchmark configuration
- Benchmark runner
- Experiment storage
- Experiment comparison

### Current Test Status

```text
44 passed
```

---

# 📊 Benchmarking

Phase 2 introduced an automated experimentation system for comparing different RAG configurations.

RAGBench can currently run experiments across:

### Chunk Size

```text
200
400
600
800
```

### Chunk Overlap

```text
0
25
50
100
```

### Top-K

```text
1
2
3
5
```

Each experiment automatically:

1. Creates the configured index
2. Runs the evaluation dataset
3. Calculates evaluation metrics
4. Measures latency
5. Stores the experiment results
6. Makes the results available for comparison

---

# 📈 Example Benchmark Results

The Phase 2 benchmark ran 10 different configurations.

| Experiment | Chunk | Overlap | Top-K | Recall | Precision | RR | Relevancy | Faithfulness | Correctness | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 400 | 50 | 3 | 1.000 | 0.400 | 0.867 | 0.763 | 0.000 | 0.734 | 1.055 |
| chunk_200 | 200 | 50 | 3 | 0.720 | 0.333 | 0.800 | 0.795 | 0.400 | 0.753 | 0.640 |
| chunk_600 | 600 | 50 | 3 | 1.000 | 0.333 | 0.800 | 0.754 | 0.000 | 0.757 | 0.811 |
| chunk_800 | 800 | 50 | 3 | 1.000 | 0.333 | 0.900 | 0.718 | 0.000 | 0.754 | 0.882 |
| overlap_0 | 400 | 0 | 3 | 1.000 | 0.400 | 0.867 | 0.788 | 0.000 | 0.691 | 0.678 |
| overlap_25 | 400 | 25 | 3 | 1.000 | 0.400 | 0.867 | 0.778 | 0.000 | 0.749 | 0.546 |
| overlap_100 | 400 | 100 | 3 | 1.000 | 0.467 | 0.867 | 0.809 | 0.200 | 0.805 | 0.690 |
| topk_1 | 400 | 50 | 1 | 0.760 | 0.800 | 0.800 | 0.804 | 0.200 | 0.770 | 0.464 |
| topk_2 | 400 | 50 | 2 | 0.800 | 0.500 | 0.800 | 0.824 | 0.000 | 0.744 | 0.525 |
| topk_5 | 400 | 50 | 5 | 1.000 | 0.240 | 0.867 | 0.769 | 0.000 | 0.790 | 0.864 |

The benchmark demonstrates that changing chunk size, overlap, and Top-K can affect retrieval quality, answer quality, and latency differently.

> **Important:** Individual metrics measure different aspects of a RAG system. A configuration that performs strongly on one metric may not perform similarly on another.

---

# 🔬 Phase 2 — Benchmarking

## Goal

> **Which RAG configuration performs differently under controlled experiments?**

Phase 2 is now complete.

### Completed

- [x] Experiment configuration
- [x] Automated experiment runner
- [x] Experiment result storage
- [x] Experiment comparison
- [x] Chunk-size experiments
- [x] Chunk-overlap experiments
- [x] Top-K experiments
- [x] Retrieval metrics
- [x] Generation metrics
- [x] Answer correctness
- [x] Latency measurement
- [x] Reference-passage evaluation
- [x] Multi-sentence reference evidence
- [x] Evidence-based Recall@K
- [x] Automated testing

### Phase 2 Result

RAGBench can now systematically run controlled RAG experiments and compare their measured retrieval, generation, correctness, and efficiency characteristics.

---

# 🧠 Evaluation Metrics

## Recall@K

RAGBench uses reference evidence to measure how much required information was retrieved within the top-K results.

```text
Recall@K =
Matched reference evidence
───────────────────────────
Total reference evidence
```

---

## Precision@K

Measures the proportion of retrieved top-K chunks that contain relevant reference evidence.

```text
Precision@K =
Relevant retrieved chunks
────────────────────────
Retrieved top-K chunks
```

---

## Reciprocal Rank

Measures the position of the first relevant retrieved result.

```text
RR = 1 / rank of first relevant result
```

A relevant result at rank 1 gives:

```text
RR = 1.0
```

---

## Mean Reciprocal Rank

MRR calculates the average Reciprocal Rank across multiple evaluation questions.

```text
MRR = Average(RR)
```

---

## Faithfulness

Faithfulness measures whether claims made by the generated answer are supported by the retrieved context.

The current implementation:

```text
Generated Answer
       ↓
Split into claims
       ↓
NLI evaluation
       ↓
Supported claims
       ↓
Faithfulness score
```

```text
Faithfulness =
Supported claims
────────────────
Total claims
```

---

## Answer Relevancy

Answer Relevancy measures semantic similarity between the question and generated answer using embeddings.

It provides a signal for whether the generated answer is addressing the question rather than drifting off-topic.

> Answer relevancy measures semantic relatedness, not factual correctness.

---

## Answer Correctness

Answer Correctness evaluates how closely the generated answer matches the expected ground-truth answer.

This provides an additional signal beyond retrieval quality and semantic relevancy.

---

## Latency

RAGBench measures the time required to process an evaluation question through the RAG pipeline.

```text
Question
   ↓
Retrieval
   ↓
Generation
   ↓
Evaluation
   ↓
Latency
```

Latency allows experiments to be compared not only by quality metrics but also by execution time.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core development |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Transformers | NLI-based evaluation |
| DeBERTa NLI | Faithfulness evaluation |
| Ollama | Local LLM inference |
| Qwen 2.5 1.5B | Answer generation |
| Scikit-learn | Similarity calculations |
| Streamlit | Evaluation dashboard |
| Pytest | Automated testing |

The project is designed to run on CPU-friendly hardware without requiring a dedicated GPU.

---

# 📁 Project Structure

```text
rag-eval-system/
│
├── data/
│   ├── corpus/
│   │   └── sample_doc.md
│   │
│   ├── eval_set/
│   │   ├── eval_set.json
│   │   └── results.json
│   │
│   └── experiments/
│       ├── baseline/
│       ├── chunk_200/
│       ├── chunk_600/
│       ├── chunk_800/
│       ├── overlap_0/
│       ├── overlap_25/
│       ├── overlap_100/
│       ├── topk_1/
│       ├── topk_2/
│       └── topk_5/
│
├── src/
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── index.py
│   │   ├── retrieve.py
│   │   └── generate.py
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py
│   │   ├── faithfulness.py
│   │   ├── answer_relevancy.py
│   │   ├── answer_correctness.py
│   │   ├── run_eval.py
│   │   └── dashboard.py
│   │
│   └── ragbench/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   └── interfaces.py
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   └── schema.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   └── passage_matching.py
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── v1_adapter.py
│       │   └── v1_benchmark.py
│       │
│       └── benchmark/
│           ├── __init__.py
│           ├── config.py
│           ├── runner.py
│           ├── storage.py
│           └── comparison.py
│
├── examples/
│   ├── __init__.py
│   ├── custom_rag/
│   ├── evaluate_v1.py
│   ├── run_experiments.py
│   └── compare_experiments.py
│
├── notebooks/
│
├── tests/
│   ├── test_ragbench_interface.py
│   ├── test_retrieval_metrics.py
│   ├── test_v1_adapter.py
│   ├── test_passage_matching.py
│   ├── test_benchmark_config.py
│   ├── test_benchmark_runner.py
│   ├── test_experiment_storage.py
│   └── test_experiment_comparison.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-eval-system
```

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If Pytest is not included:

```bash
python -m pip install pytest
```

---

# 🤖 Local LLM Setup

RAGBench uses Ollama for local answer generation.

Install Ollama and pull the model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running before executing the generation or evaluation pipeline.

---

# 📚 Build the Vector Index

Place `.md` or `.txt` documents inside:

```text
data/corpus/
```

Build the FAISS index:

```bash
python -m src.rag.index --corpus data/corpus
```

The indexing pipeline:

1. Loads documents
2. Splits documents into sentence-aware chunks
3. Generates embeddings using all-MiniLM-L6-v2
4. Normalizes embeddings
5. Builds a FAISS inner-product index
6. Stores the index and chunk metadata

---

# 🔎 Test Retrieval

Run:

```bash
python -m src.rag.retrieve
```

The retriever returns the top-K chunks along with similarity scores.

---

# 🤖 Generate an Answer

Run:

```bash
python -m src.rag.generate
```

The pipeline:

1. Accepts a question
2. Retrieves relevant chunks
3. Builds the context
4. Sends the context to Qwen
5. Generates an answer

---

# 📊 Run RAGBench Evaluation

Evaluate the original V1 RAG pipeline through the RAGBench evaluation interface:

```bash
python -m examples.evaluate_v1
```

---

# 🧪 Run the Benchmark Experiments

Phase 2 provides automated experiments for different RAG configurations.

Run:

```bash
python examples/run_experiments.py
```

The experiment runner creates separate experiment directories and stores their results.

Then compare the experiments:

```bash
python examples/compare_experiments.py
```

The comparison reports:

```text
Recall
Precision
Reciprocal Rank
Answer Relevancy
Faithfulness
Answer Correctness
Latency
```

---

# 🧩 Using RAGBench With a Custom RAG System

A custom RAG system can implement the `RAGSystem` interface.

Example:

```python
from src.ragbench.core.interfaces import RAGSystem


class MyRAG(RAGSystem):

    def retrieve(self, question, top_k=3):
        # Your retrieval implementation

        return [
            {
                "chunk_id": "doc_1",
                "text": "Relevant document content",
                "score": 0.95
            }
        ]

    def generate(self, question, retrieved_chunks):
        # Your generation implementation

        return "Generated answer"
```

The custom system can then be evaluated using:

```python
from src.ragbench.evaluation.evaluator import RAGEvaluator

rag = MyRAG()

evaluator = RAGEvaluator(rag)

results = evaluator.evaluate(
    dataset,
    top_k=3
)
```

The architecture is:

```text
Your RAG
   ↓
RAGSystem Interface
   ↓
RAGBench
   ↓
Evaluation
   ↓
Metrics
```

Your RAG system does not need to use the same internal implementation as RAGBench.

---

# 🧪 Running Tests

Run the complete test suite:

```powershell
$env:PYTHONPATH = "."
pytest -q
```

Current test status:

```text
44 passed
```

The tests verify:

- Core RAG interface
- Retrieval metrics
- Passage matching
- Reference evidence
- Evidence-based Recall
- V1 adapter
- Benchmark configuration
- Benchmark execution
- Experiment storage
- Experiment comparison

---

# 🔬 Phase 1 — Reusable Evaluation Core ✅

### Goal

> **Can RAGBench evaluate different RAG systems?**

### Completed

- [x] Framework-agnostic RAGSystem interface
- [x] Reusable RAGEvaluator
- [x] Standardized evaluation dataset
- [x] Multiple relevant reference passages
- [x] Retrieval metrics
- [x] Generation metrics
- [x] Faithfulness evaluation
- [x] Answer Relevancy
- [x] Answer Correctness
- [x] V1 RAG adapter
- [x] Custom RAG example
- [x] Automated testing

### Result

The evaluation layer is separated from the original RAG implementation.

---

# 📈 Phase 2 — Benchmarking & Experiment Comparison ✅

### Goal

> **How do different RAG configurations affect measurable performance?**

### Completed

- [x] Automated experiment configuration
- [x] Automated experiment runner
- [x] Chunk-size experiments
- [x] Chunk-overlap experiments
- [x] Top-K experiments
- [x] Experiment result storage
- [x] Experiment comparison
- [x] Retrieval metrics
- [x] Generation metrics
- [x] Answer correctness
- [x] Latency measurement
- [x] Reference evidence evaluation
- [x] Evidence-based Recall@K
- [x] 10 benchmark configurations
- [x] 44 automated tests

### Result

RAGBench can now run controlled experiments and compare different RAG configurations using multiple quality and efficiency metrics.

---

# 🔜 Phase 3 — Diagnosis & Optimization

### Goal

> **Why did my RAG fail?**

Planned features:

- [ ] Retrieval failure analysis
- [ ] Generation failure analysis
- [ ] Automatic failure classification
- [ ] Per-question diagnosis
- [ ] Failure analysis dashboard
- [ ] Reranking
- [ ] Hybrid retrieval
- [ ] Optimization experiments
- [ ] Optimization recommendations

The planned diagnostic flow is:

```text
Question
   ↓
Retrieved Context
   ↓
Retrieval Metrics
   ↓
Generated Answer
   ↓
Generation Metrics
   ↓
Failure Diagnosis
   ↓
Optimization
```

---

# 🔜 Phase 4 — Open-Source Developer Tool

### Goal

> **Make RAGBench easy for other developers to use.**

Planned:

- [ ] Command-line interface
- [ ] Improved dashboard
- [ ] Configuration-based experiments
- [ ] Python package
- [ ] Documentation
- [ ] Example integrations
- [ ] Contribution guidelines
- [ ] Open-source release

---

# 🧠 Project Vision

RAGBench is being developed around a simple idea:

> **Don't just build RAG systems. Measure them, understand them, and improve them.**

The project is evolving from a single RAG pipeline into a reusable evaluation and benchmarking framework for RAG developers.

The long-term workflow is:

```text
Does my RAG work?
        ↓
How well does it work?
        ↓
What configuration affects performance?
        ↓
Why does it fail?
        ↓
How can I improve it?
        ↓
Can the improvement be measured?
```

---

# 👨‍💻 Author

**Karthik Telukutla**

B.Tech | AI/DS | Generative AI | RAG | NLP | Python

Currently building hands-on AI/ML projects focused on RAG systems, evaluation, experimentation, and AI engineering.

---

# 📌 Project Status

```text
Current Version: V2

Phase 1 — Reusable Evaluation Core       ✅ Completed
Phase 2 — Benchmarking                   ✅ Completed
Phase 3 — Diagnosis & Optimization       🔜 Next
Phase 4 — Open-Source Developer Tool     🔜 Planned
```

### Current Test Status

```text
44 tests passed
```

---

# ⭐ If you find this project useful

Consider giving the repository a star and following the project as RAGBench evolves into a more complete RAG evaluation and optimization framework.
