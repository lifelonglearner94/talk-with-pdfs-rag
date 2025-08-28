from experiments.eval.run_eval import mrr_at_k


def test_mrr_at_k_basic():
    gold = ["PaperA.pdf", "PaperB.pdf"]
    retrieved = ["PaperC", "PaperA something", "Other"]
    # PaperA appears at rank 2 -> MRR 0.5
    assert abs(mrr_at_k(retrieved, gold) - 0.5) < 1e-6


def test_mrr_at_k_miss():
    gold = ["PaperZ.pdf"]
    retrieved = ["A", "B", "C"]
    assert mrr_at_k(retrieved, gold) == 0.0
