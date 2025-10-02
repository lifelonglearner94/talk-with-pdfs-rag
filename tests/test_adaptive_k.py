from app.core.config import RAGConfig
from app.core.retrieval_utils import choose_adaptive_k

def test_choose_adaptive_k_low_variance():
    cfg = RAGConfig(adaptive_k=True, k_min=5, k_max=15, top_k=10)
    sims = [0.80 + 1e-4*i for i in range(10)]  # very tight cluster -> expect k_min
    k = choose_adaptive_k(sims, cfg)
    assert k == cfg.k_min

def test_choose_adaptive_k_medium_variance():
    cfg = RAGConfig(adaptive_k=True, k_min=5, k_max=15, top_k=10)
    sims = [0.80, 0.75, 0.74, 0.73, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66]  # moderate std
    k = choose_adaptive_k(sims, cfg)
    assert k == cfg.top_k

def test_choose_adaptive_k_high_variance():
    cfg = RAGConfig(adaptive_k=True, k_min=5, k_max=15, top_k=10)
    sims = [0.95, 0.70, 0.40, 0.30, 0.20, 0.10, -0.05, -0.10, -0.15, -0.20]  # large spread
    k = choose_adaptive_k(sims, cfg)
    assert k == cfg.k_max
