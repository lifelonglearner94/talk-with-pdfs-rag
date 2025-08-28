from app.core.structured_splitter import StructuredPaperSplitter
from langchain_core.documents import Document


def test_token_count_not_zero_and_within_bounds():
    text = "INTRODUCTION\n" + ("A short sentence about methods and results. " * 80)
    splitter = StructuredPaperSplitter(chunk_size=400, chunk_overlap=50)
    docs = [Document(page_content=text, metadata={"source": "Dummy, 2024, Sample.pdf"})]
    chunks = splitter.split(docs)
    assert chunks, "Expected chunks"
    # token_count field must exist and be >0 and not exceed approx chunk_size/4 * 1.2 (tolerance)
    max_tokens = int(400/4 * 1.2)
    for c in chunks:
        tc = c.metadata.get('token_count')
        assert isinstance(tc, int) and tc > 0
        assert tc <= max_tokens, f"token_count {tc} exceeds tolerance {max_tokens}"
