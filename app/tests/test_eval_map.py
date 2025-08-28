from experiments.eval.run_eval import average_precision_at_k


def test_average_precision_at_k_basic():
    # Gold has 2 sources; retrieved hits at ranks 1 and 4
    retrieved = ["PaperA", "X", "Y", "PaperB"]
    gold = ["PaperA.pdf", "PaperB.pdf"]
    # AP = (1/1 + 2/4) / 2 = (1 + 0.5)/2 = 0.75
    ap = average_precision_at_k(retrieved, gold)
    assert abs(ap - 0.75) < 1e-6


def test_average_precision_at_k_miss():
    retrieved = ["A", "B", "C"]
    gold = ["Z.pdf"]
    assert average_precision_at_k(retrieved, gold) == 0.0
