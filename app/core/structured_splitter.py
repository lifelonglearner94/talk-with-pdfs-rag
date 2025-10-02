from __future__ import annotations
"""Section-aware splitter (Phase 1 initial stub).

Current implementation:
- Splits documents into paragraphs separated by blank lines.
- Detects simple section headings via regex (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion, References)
- Aggregates paragraphs under the last seen heading.
- Packs paragraphs into chunks respecting configured chunk_size with overlap in characters similar to basic splitter.

This is a *minimal* initial version to unblock further enhancement work.
"""
from typing import List, Iterable, Optional, Callable
import re
from langchain_core.documents import Document

SECTION_PATTERN = re.compile(r"^(abstract|introduction|background|methods?|methodology|results?|discussion|conclusion|conclusions|references)\b", re.IGNORECASE)
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+\S+")

class StructuredPaperSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, token_budget: Optional[int] = None, token_counter: Optional[Callable[[str], int]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Approx token budget (fallback heuristic: ~4 chars per token). Use 0.9 * size/4 to stay under limit.
        approx = max(1, int(chunk_size / 4))
        self.token_budget = token_budget or int(approx * 0.9)
        self._token_counter = token_counter or self._approx_tokens

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # Cheap heuristic avoids external deps; refine later with tiktoken if available
        return max(1, int(len(text) / 4))

    def _paragraphs(self, text: str) -> Iterable[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) == 1:
            # Fallback: further split large single paragraph into sentences to allow packing
            big = paras[0]
            # crude sentence split
            sentences = re.split(r"(?<=[.!?])\s+", big)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                yield s
            return
        for p in paras:
            if "\n" in p:
                first, rest = p.split("\n", 1)
                if self._is_heading(first) and rest.strip():
                    yield first.strip()
                    yield rest.strip()
                    continue
            yield p

    def _is_heading(self, para: str) -> bool:
        """Return True if paragraph line likely a section heading.

        Refinements:
        - Reject if contains a period anywhere (period guard) for SECTION_PATTERN matches.
        - Allow short ALLCAPS lines (3–10 tokens) without period.
        - Length ceiling 120 chars.
        """
        if len(para) > 120:
            return False
        line = para.strip()
        if not line:
            return False
        # Direct keyword style headings (avoid full sentences with periods)
        if SECTION_PATTERN.match(line.lower()) and '.' not in line:
            return True
        # ALLCAPS short heading
        if line.isupper() and '.' not in line and 3 <= len(line.split()) <= 10:
            return True
        # Numbered headings allow dotted numeric prefixes (e.g., 2.1, 3.2.1) but we must avoid
        # false positives for pure version-like or year-only lines. Accept when the line has
        # a numeric prefix followed by text (e.g. '1.1 Background'). Reject lines that are
        # only a numeric version like '1.0' or a year like '2024'.
        if NUMBERED_HEADING_PATTERN.match(line):
            first_token = line.split()[0]
            # Reject lines that begin with a 4-digit year to avoid false positives like "2024 Results show ..."
            if re.fullmatch(r"(19|20|21)\d{2}", first_token):
                return False
            # Reject if the entire line is just a numeric version (e.g., '1.0', '1.0.0'),
            # but allow numbered headings that have trailing text after the numbering.
            if re.fullmatch(r"^\d+(?:\.\d+){1,3}$", line):
                return False
            # Additional check: reject lines that look like "1.0 description..." where the text
            # after the number is lowercase/sentence-like (not proper title case).
            # Real headings typically have capitalized words after the number.
            words = line.split()
            if len(words) >= 2:
                # Check if second word (first word after number) is lowercase (not a proper heading)
                second_word = words[1]
                if second_word and second_word[0].islower():
                    return False
            # Also reject if line ends with sentence punctuation (headings typically don't end with periods)
            if line.rstrip().endswith(('.', '!', '?')):
                return False
            # Reject if line has sentence-like structure with many words (likely descriptive text, not a heading)
            # Real headings are typically short (2-6 words after number)
            if len(words) > 8:
                return False
            return True
        return False

    def _infer_section_level(self, heading: str) -> int:
        """Infer a hierarchical section level from numbering pattern.

        Examples:
        '1 Introduction' -> 1
        '1.2 Experimental Setup' -> 2
        'Results' (no numbering) -> 1 (flat default)
        """
        numbering = re.match(r"^(\d+(?:\.\d+)*)\s+", heading.strip())
        if not numbering:
            return 1
        parts = numbering.group(1).split('.')
        return min(len(parts), 6)

    def _compute_page_range(self, d: Document, content: str) -> tuple[int | None, int | None]:
        """Return (page_start, page_end) using `page_map` if present.

        `page_map` is a list of dicts: {page, start, end} giving character span for each original page
        inside the merged document text. We locate any pages whose span intersects the content span.
        """
        page_map = d.metadata.get('page_map')
        if not page_map:
            p = d.metadata.get('page')
            return p, p
        # Use last recorded chunk start offset captured during split()
        start = getattr(self, '_chunk_start', 0)
        end = start + len(content)
        covered = [pm['page'] for pm in page_map if not (pm['end'] <= start or pm['start'] >= end)]
        if not covered:
            return None, None
        return min(covered), max(covered)

    def split(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        for d in docs:
            current_section = "_preamble"
            buffer: List[str] = []
            current_len = 0
            section_index = 0
            current_tokens = 0
            current_section_level = 1
            # Track chunk start offset inside merged doc for page mapping
            doc_cursor = 0
            for para in self._paragraphs(d.page_content):
                if self._is_heading(para):
                    # Flush existing buffer before switching sections
                    if buffer:
                        content = "\n\n".join(buffer)
                        meta = dict(d.metadata)
                        # Record chunk start for page mapping
                        self._chunk_start = doc_cursor
                        page_start, page_end = self._compute_page_range(d, content)
                        meta.update({
                            "section": current_section,
                            "section_index": section_index,
                            "section_level": current_section_level,
                            "page_start": page_start,
                            "page_end": page_end,
                            "splitting_mode": "structure",
                            "token_count": self._token_counter(content),
                        })
                        out.append(Document(page_content=content, metadata=meta))
                        buffer = []
                        current_len = 0
                        current_tokens = 0
                        doc_cursor += len(content) + 2
                    current_section = para.split("\n")[0][:80]
                    section_index += 1
                    current_section_level = self._infer_section_level(current_section)
                    continue
                # Add paragraph to buffer; if exceeding chunk_size flush
                para_len = len(para)
                para_tokens = self._approx_tokens(para)
                size_exceeded = (current_len + para_len > self.chunk_size)
                token_exceeded = (current_tokens + para_tokens > self.token_budget)
                if (size_exceeded or token_exceeded) and buffer:
                    # emit chunk
                    content = "\n\n".join(buffer)
                    meta = dict(d.metadata)
                    self._chunk_start = doc_cursor
                    page_start, page_end = self._compute_page_range(d, content)
                    meta.update({
                        "section": current_section,
                        "section_index": section_index,
                        "section_level": current_section_level,
                        "page_start": page_start,
                        "page_end": page_end,
                        "splitting_mode": "structure",
                        "token_count": self._token_counter(content),
                    })
                    out.append(Document(page_content=content, metadata=meta))
                    # start new buffer with overlap (last paragraph if small)
                    overlap_text = ""
                    if buffer and self.chunk_overlap > 0:
                        last = buffer[-1]
                        if len(last) <= self.chunk_overlap:
                            overlap_text = last
                    buffer = [overlap_text] if overlap_text else []
                    current_len = len(overlap_text)
                    current_tokens = self._approx_tokens(overlap_text) if overlap_text else 0
                buffer.append(para)
                current_len += para_len + 2
                current_tokens += para_tokens
                # Hard guard: if tokens exceed budget aggressively split early
                if current_tokens > self.token_budget * 1.05 and buffer:
                    content = "\n\n".join(buffer)
                    meta = dict(d.metadata)
                    self._chunk_start = doc_cursor
                    page_start, page_end = self._compute_page_range(d, content)
                    meta.update({
                        "section": current_section,
                        "section_index": section_index,
                        "section_level": current_section_level,
                        "page_start": page_start,
                        "page_end": page_end,
                        "splitting_mode": "structure",
                        "token_count": self._token_counter(content),
                    })
                    out.append(Document(page_content=content, metadata=meta))
                    buffer = []
                    current_len = 0
                    current_tokens = 0
            # If we still have a very large final buffer (e.g., single huge paragraph) enforce final split by character size
            if buffer:
                content = "\n\n".join(p for p in buffer if p)
                meta = dict(d.metadata)
                self._chunk_start = doc_cursor
                page_start, page_end = self._compute_page_range(d, content)
                meta.update({
                    "section": current_section,
                    "section_index": section_index,
                    "section_level": current_section_level,
                    "page_start": page_start,
                    "page_end": page_end,
                    "splitting_mode": "structure",
                    "token_count": self._token_counter(content),
                })
                out.append(Document(page_content=content, metadata=meta))
        return out
