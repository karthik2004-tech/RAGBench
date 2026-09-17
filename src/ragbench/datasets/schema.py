from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationSample:
    id: str
    question: str
    ground_truth_answer: str
    reference_passages: List[str]