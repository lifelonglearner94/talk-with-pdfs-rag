from app.core.structured_splitter import StructuredPaperSplitter
from langchain_core.documents import Document


def test_heading_false_positive_guard():
    text = "Results show that the method improves performance.\n\nAnother paragraph continuing discussion."
    splitter = StructuredPaperSplitter(chunk_size=120, chunk_overlap=10)
    docs = [Document(page_content=text, metadata={"source": "Dummy, 2024, Sample.pdf"})]
    chunks = splitter.split(docs)
    # Expect section to remain _preamble (no heading detected due to period)
    assert any(c.metadata.get('section') == '_preamble' for c in chunks)


def test_basic_section_detection_and_chunking():
    text = """ABSTRACT\nThis is abstract text.\n\nIntroduction\nIntro paragraph one.\n\nMETHODS\nMethods paragraph one.\n\nResults\nResult paragraph one.\n"""
    splitter = StructuredPaperSplitter(chunk_size=120, chunk_overlap=20)
    docs = [Document(page_content=text, metadata={"source":"dummy.pdf"})]
    chunks = splitter.split(docs)
    assert chunks, "Expected chunks to be produced"
    # Ensure section metadata present
    sections = {c.metadata.get("section") for c in chunks}
    # At minimum the final section label should be present
    assert any(s for s in sections if s.lower().startswith("result")), sections
