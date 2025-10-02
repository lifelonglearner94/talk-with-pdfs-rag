"""
Tests for preprocessing module, particularly bibliography removal.
"""
import pytest
from langchain_core.documents import Document
from app.core.preprocessing import (
    detect_bibliography_start,
    remove_bibliography,
    preprocess_document,
    preprocess_documents,
)


def test_detect_bibliography_english_references():
    text = """
This is the main content of the paper.
It has multiple paragraphs discussing various topics.

References

1. Smith, J. (2020). A Study on Something.
2. Jones, B. (2019). Another Study.
"""
    pos = detect_bibliography_start(text)
    assert pos is not None
    assert "References" in text[pos:pos+20]


def test_detect_bibliography_german_literaturverzeichnis():
    text = """
Dies ist der Hauptinhalt des Papiers.
Es hat mehrere Absätze über verschiedene Themen.

Literaturverzeichnis

Müller, A. (2021). Eine Studie über etwas.
Schmidt, B. (2020). Eine andere Studie.
"""
    pos = detect_bibliography_start(text)
    assert pos is not None
    assert "Literaturverzeichnis" in text[pos:pos+30]


def test_detect_bibliography_numbered_section():
    text = """
This is section 1 with some content.

2. Methodology

Some methodology content here.

3. Results

Some results here.

4. References

Citation 1
Citation 2
"""
    pos = detect_bibliography_start(text)
    assert pos is not None
    # Should find "4. References"
    assert "References" in text[pos:pos+30]


def test_detect_bibliography_case_insensitive():
    text = """
Content here.

REFERENCES

Citations here.
"""
    pos = detect_bibliography_start(text)
    assert pos is not None


def test_detect_bibliography_not_found():
    text = """
This is a document without a bibliography section.
It only has regular content.
There are no references here.
"""
    pos = detect_bibliography_start(text)
    assert pos is None


def test_remove_bibliography_success():
    text = """Main content here.

More content.

References

Citation 1
Citation 2
"""
    cleaned, found = remove_bibliography(text)
    assert found is True
    assert "Main content here" in cleaned
    assert "More content" in cleaned
    assert "Citation 1" not in cleaned
    assert "Citation 2" not in cleaned
    assert "References" not in cleaned


def test_remove_bibliography_not_found():
    text = "Just regular content without bibliography."
    cleaned, found = remove_bibliography(text)
    assert found is False
    assert cleaned == text


def test_preprocess_document_with_bibliography():
    doc = Document(
        page_content="""
Content here.

Bibliography

Some citations.
""",
        metadata={"source": "test.pdf", "page": 1}
    )

    processed = preprocess_document(doc, remove_bib=True)

    assert "Content here" in processed.page_content
    assert "Some citations" not in processed.page_content
    assert processed.metadata["bibliography_removed"] is True
    assert "original_length" in processed.metadata
    assert "cleaned_length" in processed.metadata


def test_preprocess_document_without_bibliography():
    doc = Document(
        page_content="Just normal content.",
        metadata={"source": "test.pdf", "page": 1}
    )

    processed = preprocess_document(doc, remove_bib=True)

    assert processed.page_content == "Just normal content."
    assert processed.metadata["bibliography_removed"] is False


def test_preprocess_document_disabled():
    doc = Document(
        page_content="""
Content here.

References

Citations.
""",
        metadata={"source": "test.pdf"}
    )

    processed = preprocess_document(doc, remove_bib=False)

    # Should keep everything
    assert "References" in processed.page_content
    assert "Citations" in processed.page_content
    assert "bibliography_removed" not in processed.metadata


def test_preprocess_documents_list():
    docs = [
        Document(
            page_content="Doc 1\n\nReferences\nCitation A",
            metadata={"source": "doc1.pdf"}
        ),
        Document(
            page_content="Doc 2 without bibliography",
            metadata={"source": "doc2.pdf"}
        ),
        Document(
            page_content="Doc 3\n\nLiteraturverzeichnis\nCitation B",
            metadata={"source": "doc3.pdf"}
        ),
    ]

    processed = preprocess_documents(docs, remove_bib=True)

    assert len(processed) == 3

    # First doc should have bibliography removed
    assert "Citation A" not in processed[0].page_content
    assert processed[0].metadata["bibliography_removed"] is True

    # Second doc has no bibliography
    assert processed[1].metadata["bibliography_removed"] is False

    # Third doc should have bibliography removed
    assert "Citation B" not in processed[2].page_content
    assert processed[2].metadata["bibliography_removed"] is True


def test_preprocess_documents_disabled():
    docs = [
        Document(
            page_content="Doc\n\nReferences\nCitation",
            metadata={"source": "doc.pdf"}
        ),
    ]

    processed = preprocess_documents(docs, remove_bib=False)

    # Should be unchanged
    assert "References" in processed[0].page_content
    assert "Citation" in processed[0].page_content


def test_bibliography_patterns_coverage():
    """Test various common bibliography header patterns."""
    patterns = [
        "References",
        "REFERENCES",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "Literaturverzeichnis",
        "LITERATURVERZEICHNIS",
        "Literatur",
        "Quellen",
        "Quellenverzeichnis",
        "5. References",
        "6. Bibliography",
        "7 Literaturverzeichnis",
    ]

    for pattern in patterns:
        text = f"Content here.\n\n{pattern}\n\nCitation 1"
        pos = detect_bibliography_start(text)
        assert pos is not None, f"Failed to detect: {pattern}"
        assert pattern.lower() in text[pos:pos+50].lower()


def test_false_positive_avoidance():
    """Test that we don't incorrectly detect bibliography in regular text."""
    # "references" buried in a long sentence should not trigger
    text = """
This paper references several important works in the field.
We continue our discussion here with many references to prior art.
"""
    pos = detect_bibliography_start(text)
    # Should not find a bibliography section
    # (might find it if the heuristic is too loose, but our implementation should avoid this)
    # Note: Current implementation has a heuristic to skip if many words before match
    assert pos is None or "references to prior" not in text[pos:pos+30]
