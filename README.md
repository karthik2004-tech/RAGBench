# RAG Evaluation System

A from-scratch evaluation harness for Retrieval-Augmented Generation (RAG) pipelines.
Scores **retrieval quality** (Recall@k, Precision@k, MRR) and **generation quality**
(faithfulness via NLI entailment, answer relevancy via embedding similarity) —
fully local, fully free, no API keys required.

Inspired by the methodology behind [RAGAS](https://github.com/explodinggradients/ragas)
and [open-rag-eval](https://github.com/vectara/open-rag-eval), reimplemented from first
principles to demonstrate the underlying mechanics rather than depend on them as a
black box.

## Why this exists

Most RAG portfolio projects stop at "I built retrieval + generation." This project
answers the question that actually matters in production: **how do you know it's
working?** It ships a labeled evaluation set, a metrics engine, and a dashboard to
inspect per-query scores and flag hallucinated answers.

## Architecture

```
Query
  │
  ├─► Embedding (MiniLM, CPU) ─► FAISS retrieval ─► top-k chunks
  │                                                     │
  │                                                     ▼
  └─────────────────────────────────► Local LLM (Ollama) ─► Answer
                                                             │
                                                             ▼
                                          Eval Harness (this project)
                                    ┌────────────┬────────────┬────────────┐
                                    Recall@k    Faithfulness  Answer
                                    Precision@k (NLI entail.) Relevancy
                                    MRR
```

## Stack (100% local, 100% free)

| Component        | Tool                                   | Why |
|-------------------|-----------------------------------------|-----|
| Embeddings        | `sentence-transformers` (all-MiniLM-L6-v2) | ~80MB, CPU-fast |
| Vector store       | FAISS (CPU)                             | No GPU dependency |
| Generation         | Ollama + `qwen2.5:1.5b` (or `phi3:mini`) | Runs on 4GB VRAM or CPU |
| Faithfulness check | `cross-encoder/nli-deberta-v3-small`     | ~140MB, CPU-fast NLI |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# pull a small local model (one-time, ~1GB download)
ollama pull qwen2.5:1.5b
```

## Quickstart

```bash
# 1. Index your corpus
python -m src.rag.index --corpus data/corpus

# 2. Run the eval harness against your labeled eval set
python -m src.eval.run_eval --eval-set data/eval_set/eval_set.json

# 3. Launch the results dashboard
streamlit run src/eval/dashboard.py
```

## Repo structure

```
rag-eval-system/
├── data/
│   ├── corpus/           # your source documents (txt/md)
│   └── eval_set/          # hand-labeled question/answer/ground-truth-chunk pairs
├── src/
│   ├── rag/
│   │   ├── index.py       # chunking + embedding + FAISS index build
│   │   ├── retrieve.py     # top-k retrieval
│   │   └── generate.py     # Ollama-based answer generation
│   └── eval/
│       ├── retrieval_metrics.py   # Recall@k, Precision@k, MRR
│       ├── faithfulness.py        # claim decomposition + NLI entailment
│       ├── answer_relevancy.py    # embedding cosine similarity
│       ├── run_eval.py            # orchestrates a full eval run
│       └── dashboard.py           # Streamlit results viewer
├── notebooks/             # exploration / metric sanity checks
├── tests/                 # unit tests for each metric
└── requirements.txt
```

## Eval set format

See `data/eval_set/eval_set.json` for the schema. Each entry is a hand-curated
question with its ground-truth source chunk — the manual curation step is itself
part of the project (dataset construction is half of real eval work).

## Roadmap / stretch goals

- [ ] Ablation runner: sweep chunk size (256/384/512) and top-k, plot Recall@k curves
- [ ] Compare two retrieval configs side-by-side in the dashboard
- [ ] Add reranking stage and measure its effect on precision
