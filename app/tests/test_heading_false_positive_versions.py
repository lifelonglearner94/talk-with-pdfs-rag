from app.core.structured_splitter import StructuredPaperSplitter
from langchain_core.documents import Document


def test_version_number_not_heading():
    # Lines like 1.0 or 2.0 are often version numbers / figure labels, not section headings.
    text = "1.0 This line describes version info not a heading.\n\nSome following paragraph with details."
    splitter = StructuredPaperSplitter(chunk_size=160, chunk_overlap=10)
    docs = [Document(page_content=text, metadata={"source": "Dummy, 2024, Sample.pdf"})]
    chunks = splitter.split(docs)
    # Expect section to remain _preamble (no heading captured)
    assert any(c.metadata.get('section') == '_preamble' for c in chunks)


def test_year_prefixed_line_not_heading():
    text = "2024 Study results show improvements.\n\nAnother paragraph."
    splitter = StructuredPaperSplitter(chunk_size=160, chunk_overlap=10)
    docs = [Document(page_content=text, metadata={"source": "Dummy, 2024, Sample.pdf"})]
    chunks = splitter.split(docs)
    assert any(c.metadata.get('section') == '_preamble' for c in chunks)
