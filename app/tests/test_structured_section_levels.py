from app.core.structured_splitter import StructuredPaperSplitter
from langchain_core.documents import Document


def test_numbered_heading_section_levels():
    text = """1 Introduction\nThis is intro.\n\n1.1 Background\nDetails background paragraph.\n\n2 Methods\nMethod paragraph."""
    splitter = StructuredPaperSplitter(chunk_size=180, chunk_overlap=10)
    docs = [Document(page_content=text, metadata={"source": "Sample, 2024, Test.pdf"})]
    chunks = splitter.split(docs)
    assert chunks, "No chunks produced"
    levels = {}
    for c in chunks:
        sec = c.metadata.get("section")
        lvl = c.metadata.get("section_level")
        if sec and lvl:
            levels.setdefault(sec, lvl)
    # Check expectations
    assert levels.get("1 Introduction") == 1
    # Either the chunk captured exact heading or overshadowed; guard
    assert any(lvl == 2 for sec, lvl in levels.items() if sec.startswith("1.1")), levels
