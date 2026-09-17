from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer("all-MiniLM-L6-v2")


def answer_correctness_score(
    generated_answer: str,
    ground_truth_answer: str,
) -> float:

    embeddings = _model.encode(
        [generated_answer, ground_truth_answer],
        convert_to_numpy=True,
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    return float(score)