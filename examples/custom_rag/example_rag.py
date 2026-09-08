from typing import Any, Dict, List

from src.ragbench.core.interfaces import RAGSystem


class ExampleRAG(RAGSystem):
    """
    Simple example RAG system.

    This represents an external developer's RAG system.
    """

    def __init__(self):
        self.documents = [
            {
                "chunk_id": "doc_1",
                "text": (
                    "Radar sensors detect the distance and "
                    "speed of nearby objects."
                ),
            },
            {
                "chunk_id": "doc_2",
                "text": (
                    "LiDAR uses laser pulses to create "
                    "a detailed representation of the environment."
                ),
            },
            {
                "chunk_id": "doc_3",
                "text": (
                    "Ultrasonic sensors are commonly used "
                    "for short-range detection."
                ),
            },
        ]

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        question_lower = question.lower()

        results = []

        keywords = {
            "radar": ["radar", "sensors", "sensor"],
            "lidar": ["lidar", "sensors", "sensor"],
            "ultrasonic": ["ultrasonic", "sensors", "sensor"],
        }

        for document in self.documents:
            text_lower = document["text"].lower()

            score = 0.0

            for sensor, words in keywords.items():
                if sensor in text_lower:
                    matches = sum(1 for word in words if word in question_lower)

                    if matches > 0:
                        score = max(score, matches / len(words))

            if score > 0:
                results.append(
                    {
                        "chunk_id": document["chunk_id"],
                        "text": document["text"],
                        "score": score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:

        if not retrieved_chunks:

            return "I don't know based on the given context."

        context = " ".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        return (
            "Based on the retrieved context: "
            + context
        )