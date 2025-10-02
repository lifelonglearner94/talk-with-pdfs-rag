"""
Preprocessing utilities for PDF documents, including bibliography removal.
"""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document
from .logging import logger


# Common bibliography section headers in multiple languages
BIBLIOGRAPHY_PATTERNS = [
    # English
    r'\breferences\b',
    r'\bbibliography\b',
    r'\bworks\s+cited\b',
    r'\bliterature\s+cited\b',
    # German
    r'\bliteraturverzeichnis\b',
    r'\bliteratur\b',
    r'\bquellen\b',
    r'\bquellenverzeichnis\b',
    # Generic numbered sections that might be bibliography
    r'\b\d+\.?\s+(?:references|bibliography|literaturverzeichnis|literatur)\b',
]


def detect_bibliography_start(text: str) -> int | None:
    """
    Detect the start position of the bibliography section in text.

    Returns the character position where bibliography starts, or None if not found.
    Uses heuristics based on common section headers and formatting patterns.
    """
    # Compile patterns with case-insensitive flag
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in BIBLIOGRAPHY_PATTERNS]

    # Look for bibliography headers, typically at the start of a line
    lines = text.split('\n')

    for i, line in enumerate(lines):
        # Check if line contains a bibliography header
        line_stripped = line.strip()

        # Skip very short lines (likely false positives)
        if len(line_stripped) < 5:
            continue

        # Check against all patterns
        for pattern in patterns:
            match = pattern.search(line_stripped)
            if match:
                # Additional heuristic: the header should be relatively prominent
                # (i.e., not buried in a long sentence)
                words_before_match = line_stripped[:match.start()].split()

                # If there are many words before the match, it's probably not a section header
                if len(words_before_match) > 5:
                    continue

                # Additional check: the line should be relatively short (a header, not prose)
                # Typical headers are < 50 chars
                if len(line_stripped) > 50:
                    continue

                # Check if this is the only substantial text on the line (or mostly so)
                # to avoid matching "There are no references here." style sentences
                words_in_line = line_stripped.split()
                # If there are more than 4-5 words and the match is not prominent, skip
                if len(words_in_line) > 5 and match.start() > len(line_stripped) * 0.3:
                    continue

                # Additional check: exclude lines that end with common sentence punctuation
                # (suggesting it's prose, not a header)
                if line_stripped.endswith(('.', '?', '!', ',')):
                    continue

                # Check if line starts with "There" or other common sentence starters
                # that indicate it's prose, not a header
                prose_starters = ['there', 'this', 'these', 'those', 'it', 'we', 'they', 'the', 'a', 'an']
                first_word = words_in_line[0].lower() if words_in_line else ''
                if first_word in prose_starters:
                    continue

                # Check that the match is near the start (within first 30% of line)
                # to avoid "Doc 2 without bibliography" type false positives
                if len(words_before_match) > 2 or match.start() > len(line_stripped) * 0.3:
                    continue

                # Calculate approximate character position in original text
                # This is approximate because of line splitting
                char_position = sum(len(lines[j]) + 1 for j in range(i))

                logger.debug(
                    "Bibliography detected at line %d (char ~%d): %s",
                    i, char_position, line_stripped[:60]
                )
                return char_position

    return None
def remove_bibliography(text: str) -> tuple[str, bool]:
    """
    Remove bibliography section from text.

    Returns:
        tuple: (cleaned_text, was_bibliography_found)
    """
    bib_start = detect_bibliography_start(text)

    if bib_start is None:
        return text, False

    # Keep everything before the bibliography
    cleaned_text = text[:bib_start].strip()

    # Log information about what was removed
    removed_length = len(text) - len(cleaned_text)
    removed_pct = (removed_length / len(text) * 100) if text else 0

    logger.debug(
        "Bibliography removed: %d chars (%.1f%% of document)",
        removed_length, removed_pct
    )

    return cleaned_text, True


def preprocess_document(doc: Document, remove_bib: bool = True) -> Document:
    """
    Preprocess a single document.

    Args:
        doc: The document to preprocess
        remove_bib: Whether to remove bibliography sections

    Returns:
        A new Document with preprocessing applied
    """
    content = doc.page_content
    metadata = doc.metadata.copy()

    if remove_bib:
        cleaned_content, bib_found = remove_bibliography(content)
        metadata['bibliography_removed'] = bib_found

        if bib_found:
            metadata['original_length'] = len(content)
            metadata['cleaned_length'] = len(cleaned_content)
            content = cleaned_content

    return Document(page_content=content, metadata=metadata)


def preprocess_documents(docs: List[Document], remove_bib: bool = True) -> List[Document]:
    """
    Preprocess a list of documents.

    Args:
        docs: List of documents to preprocess
        remove_bib: Whether to remove bibliography sections

    Returns:
        List of preprocessed documents
    """
    if not remove_bib:
        logger.debug("Bibliography removal disabled, skipping preprocessing")
        return docs

    logger.info("Preprocessing %d documents (bibliography removal enabled)", len(docs))

    processed_docs = []
    bib_removed_count = 0

    for doc in docs:
        processed_doc = preprocess_document(doc, remove_bib=remove_bib)
        processed_docs.append(processed_doc)

        if processed_doc.metadata.get('bibliography_removed', False):
            bib_removed_count += 1

    logger.info(
        "Preprocessing complete: bibliography removed from %d/%d documents",
        bib_removed_count, len(docs)
    )

    return processed_docs
