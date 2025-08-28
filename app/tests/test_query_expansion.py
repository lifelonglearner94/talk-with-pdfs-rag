from app.core.query_expansion import generate_expansions


def test_generate_expansions_synonym_and_decomp():
    q = "Kubernetes performance and scalability"
    ex = generate_expansions(q, max_new=5)
    # Expect at least one synonym replacement and one decomposition fragment
    assert any("k8s" in e for e in ex), f"Expected synonym variant in {ex}"
    assert any(e.strip().startswith("kubernetes performance") or e.startswith("performance and scalability") for e in ex) or any("performance" in e and "scalability" not in e for e in ex)
    assert len(ex) <= 5


def test_generate_expansions_dedup():
    q = "alpha and beta, alpha"
    ex = generate_expansions(q, max_new=5)
    # Ensure no duplicates
    assert len(ex) == len(set(ex))
