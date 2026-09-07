from src.eval.retrieval_metrics import recall_at_k, precision_at_k, reciprocal_rank, mean_reciprocal_rank


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], "b", k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["a", "b", "c"], "z", k=3) == 0.0


def test_recall_at_k_outside_window():
    assert recall_at_k(["a", "b", "c"], "c", k=2) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], "b", k=3) == 1 / 3


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["a", "b", "c"], "c") == 1 / 3


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0


def test_mean_reciprocal_rank():
    all_retrieved = [["a", "b"], ["x", "y"]]
    all_ground_truth = ["a", "y"]
    # first query: rank 1 -> RR=1.0; second query: rank 2 -> RR=0.5
    assert mean_reciprocal_rank(all_retrieved, all_ground_truth) == 0.75
