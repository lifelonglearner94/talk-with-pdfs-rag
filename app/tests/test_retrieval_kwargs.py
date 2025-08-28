from app.core import RAGConfig
from app.core.retrieval_utils import build_search_kwargs


def test_similarity_kwargs():
    cfg = RAGConfig(retrieval_strategy='similarity', top_k=5)
    assert build_search_kwargs(cfg) == {"k": 5}


def test_mmr_kwargs_fetch_k_min_floor():
    cfg = RAGConfig(retrieval_strategy='mmr', top_k=5, mmr_fetch_k_factor=2, mmr_min_fetch_k=30)
    kw = build_search_kwargs(cfg)
    # fetch_k should respect min floor (5*2=10 < 30)
    assert kw['fetch_k'] == 30
    assert kw['k'] == 5
    assert 'lambda_mult' in kw


def test_mmr_kwargs_fetch_k_factor():
    cfg = RAGConfig(retrieval_strategy='mmr', top_k=20, mmr_fetch_k_factor=4, mmr_min_fetch_k=10)
    kw = build_search_kwargs(cfg)
    # 20*4=80 > min
    assert kw['fetch_k'] == 80
