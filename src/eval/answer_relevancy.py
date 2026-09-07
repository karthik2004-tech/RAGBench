"""
Answer relevancy: does the generated answer actually address the question asked?

Method: embed both the question and the answer, score cosine similarity.
A low score often means the model drifted off-topic or answered a different
question than the one retrieved context was meant to support.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def answer_relevancy_score(question: str, answer: str) -> float:
    model = _get_model()
    embeddings = model.encode([question, answer], convert_to_numpy=True)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(sim), 3)


if __name__ == "__main__":
    q = "What does RAG combine?"
    a = "RAG combines a retriever and a generator to ground answers in real documents."
    print(f"Answer relevancy: {answer_relevancy_score(q, a)}")
