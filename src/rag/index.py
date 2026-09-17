"""
Build a FAISS index over a local text corpus.

Usage:
    python -m src.rag.index --corpus data/corpus --out data/index
"""
import argparse
import json
import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into sentence-aware chunks.

    chunk_size is the target number of characters.
    Sentences are kept intact whenever possible.

    overlap is applied using complete sentences
    from the previous chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # Split text into sentences while preserving
    # sentence boundaries.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # If adding this sentence exceeds the target
        # and we already have content, finalize the chunk.
        if (
            current_sentences
            and current_length + sentence_length + 1
            > chunk_size
        ):
            chunks.append(
                " ".join(current_sentences).strip()
            )

            # Build overlap using complete previous
            # sentences.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(
                current_sentences
            ):
                if (
                    overlap_length
                    + len(previous_sentence)
                    + 1
                    <= overlap
                ):
                    overlap_sentences.insert(
                        0,
                        previous_sentence,
                    )
                    overlap_length += (
                        len(previous_sentence) + 1
                    )
                else:
                    break

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    # Add the final chunk.
    if current_sentences:
        chunks.append(
            " ".join(current_sentences).strip()
        )

    return chunks


def load_corpus(
    corpus_dir: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Read every .md/.txt file and chunk it."""

    records = []

    for fname in sorted(os.listdir(corpus_dir)):

        if not fname.endswith((".md", ".txt")):
            continue

        path = os.path.join(corpus_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = os.path.splitext(fname)[0]

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk,
            })

    return records


def build_index(corpus_dir: str, out_dir: str, model_name: str = "all-MiniLM-L6-v2",chunk_size: int = 400, chunk_overlap: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    records = load_corpus(corpus_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()
    build_index(args.corpus, args.out, args.model,args.chunk_size,args.chunk_overlap)
