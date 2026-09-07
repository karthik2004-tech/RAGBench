"""
Build a FAISS index over a local text corpus.

Usage:
    python -m src.rag.index --corpus data/corpus --out data/index
"""
import argparse
import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Simple fixed-size character chunker with overlap.

    Swap this out for a sentence/paragraph-aware chunker as a later
    improvement — this version is deliberately simple so the effect of
    chunk size on retrieval quality (see roadmap: ablation runner) is
    easy to reason about.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_corpus(corpus_dir: str) -> list[dict]:
    """Read every .md/.txt file in corpus_dir and chunk it."""
    records = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith((".md", ".txt")):
            continue
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = os.path.splitext(fname)[0]
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk,
            })
    return records


def build_index(corpus_dir: str, out_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)

    records = load_corpus(corpus_dir)
    if not records:
        raise ValueError(f"No .md/.txt files found in {corpus_dir}")

    print(f"Loaded {len(records)} chunks from {corpus_dir}")

    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "records.pkl"), "wb") as f:
        pickle.dump(records, f)

    print(f"Index built: {index.ntotal} vectors, dim {dim}. Saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="data/index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_index(args.corpus, args.out, args.model)
