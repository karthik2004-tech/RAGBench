from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvaluationSample:
    """
    one question used to evaluate a RAG system
    """

    id: str
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: List[str]