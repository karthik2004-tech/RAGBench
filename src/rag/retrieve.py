"""
Retrieve top-k chunks for a query against a pre-built FAISS index.
"""
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_dir: str = "data/index", model_name: str = "all-MiniLM-L6-v2"):
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "records.pkl"), "rb") as f:
            self.records = pickle.load(f)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.records[idx]
            results.append({
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "text": record["text"],
                "score": float(score),
            })
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What does RAG combine?"
    for r in retriever.retrieve(query, top_k=3):
        print(f"[{r['score']:.3f}] {r['chunk_id']}: {r['text'][:80]}...")
