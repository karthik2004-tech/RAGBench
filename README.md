# RAGBench — RAG Evaluation & Optimization Framework

RAGBench is an open-source framework for evaluating and benchmarking Retrieval-Augmented Generation (RAG) systems.

The project started as a RAG pipeline with built-in evaluation and is now evolving into a reusable, framework-agnostic evaluation platform that can be used with different RAG implementations.

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
- Optimize RAG pipelines based on measurable results

---

## 🏗️ Architecture

### V1 — Baseline RAG Pipeline

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

### V2 Phase 1 — Reusable Evaluation Core

The major change in Phase 1 is separating the evaluation framework from a specific RAG implementation.

```text
                 Any RAG System
                       ↓
                 RAGSystem API
                       ↓
                  RAGEvaluator
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
   Retrieval Metrics        Generation Metrics
          ↓                         ↓
 Recall / Precision        Faithfulness / Relevancy
          └────────────┬────────────┘
                       ↓
                Evaluation Results
```

This allows RAGBench to evaluate different RAG systems without requiring them to use the same retriever, embedding model, vector search engine, or LLM.

---

## ✨ Features

### 🔎 Retrieval Evaluation

RAGBench currently supports:

- Recall@K
- Precision@K
- Reciprocal Rank
- Mean Reciprocal Rank (MRR)
- Multiple relevant chunks per evaluation question

### 🤖 Generation Evaluation

- Faithfulness evaluation using Natural Language Inference (NLI)
- Answer Relevancy using semantic similarity
- Per-claim faithfulness analysis

### 🧩 Framework-Agnostic RAG Interface

RAG systems can implement the standard:

```python
class RAGSystem:
    retrieve(...)
    generate(...)
    answer(...)
```

This creates a common interface between external RAG systems and RAGBench.

### 🔌 Adapters

Existing RAG pipelines can be connected to RAGBench through adapters. The project currently includes an adapter for the original V1 RAG pipeline.

### 🧪 Automated Testing

The project includes automated tests covering:

- RAG system interface
- Retrieval metrics
- Multiple relevant chunks
- V1 adapter
- Framework integration

**Current status: 12 tests passed**

---

## 📊 Example Evaluation

For an ADAS-related question, the current V1 pipeline produced:

| Metric | Score |
|---|---|
| Recall@3 | 1.000 |
| Precision@3 | 0.667 |
| Reciprocal Rank | 1.000 |
| Faithfulness | 0.000 |
| Answer Relevancy | 0.803 |

**Retrieved chunks:**

1. `sample_doc_chunk_3`
2. `sample_doc_chunk_4`
3. `sample_doc_chunk_2`

The evaluation shows that the relevant information was successfully retrieved, while the generation metrics provide additional signals about the quality of the final answer.

> **Note:** Faithfulness is currently implemented using a local NLI-based heuristic, so the score should be interpreted as an evaluation signal rather than an absolute measure of factual correctness.

---

## 🛠️ Tech Stack

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

## 📁 Project Structure

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
│   └── index/
│       ├── index.faiss
│       └── records.pkl
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
│       │   └── evaluator.py
│       │
│       └── adapters/
│           ├── __init__.py
│           └── v1_adapter.py
│
├── examples/
│   ├── __init__.py
│   ├── custom_rag/
│   │   ├── __init__.py
│   │   └── example_rag.py
│   └── evaluate_v1.py
│
├── notebooks/
│
├── tests/
│   ├── test_ragbench_interface.py
│   ├── test_retrieval_metrics.py
│   └── test_v1_adapter.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-eval-system
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If Pytest is not included in the requirements:

```bash
python -m pip install pytest
```

---

## 🤖 Local LLM Setup

RAGBench uses Ollama for local answer generation.

Install Ollama and pull the model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running before executing the generation or evaluation pipeline.

---

## 📚 Build the Vector Index

Place `.md` or `.txt` documents inside:

```text
data/corpus/
```

Then build the FAISS index:

```bash
python -m src.rag.index --corpus data/corpus
```

This creates:

```text
data/index/
├── index.faiss
└── records.pkl
```

The indexing pipeline:

1. Loads documents
2. Splits documents into overlapping chunks
3. Generates embeddings using all-MiniLM-L6-v2
4. Normalizes embeddings
5. Builds a FAISS inner-product index
6. Stores the index and chunk metadata

---

## 🔎 Test Retrieval

Run:

```bash
python -m src.rag.retrieve
```

The retriever returns the top-K chunks along with similarity scores.

---

## 🤖 Generate an Answer

Run:

```bash
python -m src.rag.generate
```

The pipeline:

1. Accepts a question
2. Retrieves relevant chunks
3. Builds the context
4. Sends the context to Qwen
5. Generates a grounded answer

---

## 📊 Run RAGBench Evaluation

The original V1 RAG pipeline can be evaluated through the V2 RAGBench interface.

Run:

```bash
python -m examples.evaluate_v1
```

Example output:

```text
===== RAGBench V2 Evaluation =====

Question: What sensors are commonly used in ADAS (Advanced Driver Assistance Systems) to detect nearby objects?

Recall@3: 1.000
Precision@3: 0.667
Reciprocal Rank: 1.000
Faithfulness: 0.000
Answer Relevancy: 0.803

Generated Answer:
Radar sensors and ultrasonic sensors are commonly used in ADAS to detect nearby objects.

Retrieved Chunks:
- sample_doc_chunk_3 (score=0.725)
- sample_doc_chunk_4 (score=0.670)
- sample_doc_chunk_2 (score=0.225)
```

---

## 🧩 Using RAGBench With a Custom RAG System

The main purpose of V2 Phase 1 is to allow developers to evaluate different RAG implementations.

A custom RAG system can implement the `RAGSystem` interface.

**Example:**

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

The important concept is:

```text
Your RAG
    ↓
RAGSystem Interface
    ↓
RAGBench
    ↓
Evaluation
```

Your RAG system does not need to use the same internal implementation as RAGBench.

---

## 🧪 Running Tests

Run the complete test suite:

```bash
python -m pytest
```

**Current Phase 1 status: 12 passed**

The tests verify the core interface, evaluation metrics, adapter integration, and framework behavior.

---

## 🧠 Evaluation Metrics

### Recall@K

Measures how many of the relevant chunks were successfully retrieved within the top-K results.

```text
Recall@K = Relevant chunks retrieved in Top-K / Total relevant chunks
```

Higher is generally better.

### Precision@K

Measures how many retrieved chunks in the top-K results were actually relevant.

```text
Precision@K = Relevant chunks in Top-K / Total chunks retrieved in Top-K
```

Higher is generally better.

### Reciprocal Rank

Measures how high the first relevant chunk appears in the ranked retrieval results.

```text
RR = 1 / rank of first relevant result
```

A relevant result appearing at rank 1 gives `RR = 1.0`.

### Mean Reciprocal Rank (MRR)

MRR calculates the average reciprocal rank across multiple evaluation questions. Higher values indicate that relevant information tends to appear near the top of the retrieval results.

### Faithfulness

Faithfulness measures whether claims made by the generated answer are supported by the retrieved context.

The current implementation:

1. Splits the generated answer into claims
2. Uses an NLI model to evaluate each claim against the retrieved context
3. Calculates the fraction of supported claims

```text
Faithfulness = Supported claims / Total claims
```

### Answer Relevancy

Answer Relevancy measures semantic similarity between the question and generated answer using embeddings. It provides a signal for whether the generated answer is addressing the question rather than drifting off-topic.

> Answer relevancy measures semantic relatedness, not factual correctness.

---

## 🔬 V2 Phase 1

**Goal:** Can RAGBench evaluate different RAG systems?

Phase 1 focuses on creating a reusable evaluation core rather than optimizing a single RAG pipeline.

**Completed:**

- [x] Framework-agnostic RAGSystem interface
- [x] Reusable RAGEvaluator
- [x] Standardized evaluation dataset
- [x] Support for multiple relevant chunks
- [x] Recall@K
- [x] Precision@K
- [x] Reciprocal Rank
- [x] MRR
- [x] Faithfulness evaluation
- [x] Answer Relevancy
- [x] V1 RAG adapter
- [x] Custom RAG example
- [x] Automated tests

**Phase 1 Result:** The evaluation layer is now separated from the original RAG implementation. This means RAGBench can evaluate a RAG system through a common interface instead of requiring the system to use the original V1 implementation.

---

## 📈 Roadmap

### Phase 1 — Reusable Evaluation Core ✅

**Goal:** Can RAGBench evaluate different RAG systems?

Completed:
- Framework-agnostic RAG interface
- Reusable evaluation engine
- Standardized evaluation dataset
- Multiple relevant chunks
- Retrieval metrics
- Generation metrics
- V1 adapter
- Custom RAG example
- Automated testing

### Phase 2 — Benchmarking & Experiment Comparison 🔄

**Goal:** Which RAG configuration performs better?

Planned experiments:
- Chunk size
- Chunk overlap
- Top-K
- Embedding models
- Prompt configurations
- LLM configurations

The goal is to run controlled experiments and compare the resulting evaluation metrics.

### Phase 3 — Diagnosis & Optimization 🔜

**Goal:** Why did my RAG fail?

Planned features:
- Retrieval failure analysis
- Generation failure analysis
- Latency measurement
- Reranking
- Hybrid retrieval
- Optimization recommendations

### Phase 4 — Open-Source Developer Tool 🔜

**Goal:** Make RAGBench easy for other developers to use.

Planned:
- Command-line interface
- Improved dashboard
- Configuration-based experiments
- Python package
- Documentation
- Example integrations
- Contribution guidelines
- Open-source release

---

## 🎯 Project Vision

RAGBench is being developed around a simple idea:

> Don't just build RAG systems. Measure them, understand them, and improve them.

The project is evolving from a single RAG pipeline into a reusable evaluation and benchmarking framework for RAG developers. The long-term vision is to help developers answer:

```text
Does my RAG work?
       ↓
How well does it work?
       ↓
Why does it fail?
       ↓
Which configuration is better?
       ↓
How can I improve it?
```

---

## 👨‍💻 Author

**karthik telukutla**

B.Tech | AI/DS | Generative AI | RAG | NLP | Python

Currently building hands-on AI/ML projects focused on RAG systems, evaluation, experimentation, and AI engineering.

---

## 📌 Project Status

- **Current Version:** V2 — Phase 1
- **Status:** Phase 1 Completed
- **Tests:** 12 Passed
- **Next Milestone:** Phase 2 — Benchmarking & Experiment Comparison

---

## ⭐ If you find this project useful

Consider giving the repository a star and following the project as RAGBench evolves into a more complete RAG evaluation and optimization framework.
