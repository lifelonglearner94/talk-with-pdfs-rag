from __future__ import annotations
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .config import RAGConfig
from .structured_splitter import StructuredPaperSplitter
try:  # optional dependency
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAS_TIKTOKEN = False

class DocumentSplitter:
    """Facade selecting underlying splitting strategy based on config.

    Existing call sites instantiate with numeric params; keep backward compatible
    signature while allowing optional config injection.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, config: RAGConfig | None = None):
        self.config = config
        self._basic = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )
        token_counter = None
        if config and getattr(config, 'accurate_token_count', False) and _HAS_TIKTOKEN:
            # Choose encoding: explicit override or derive from embedding model hint
            enc_name = config.token_encoding_name or 'cl100k_base'
            try:  # pragma: no cover - depends on external lib
                enc = tiktoken.get_encoding(enc_name)
                token_counter = lambda text: len(enc.encode(text))  # noqa: E731
            except Exception:  # fallback silently
                token_counter = None
        self._structured = StructuredPaperSplitter(chunk_size, chunk_overlap, token_counter=token_counter)

    def split(self, docs: List[Document]) -> List[Document]:
        mode = (self.config.chunking_mode if self.config else "basic")
        if mode == "structure":
            # Group per source so structured splitter can compute page ranges across pages
            grouped: Dict[str, List[Document]] = {}
            for d in docs:
                src = d.metadata.get("source", "__unknown__")
                grouped.setdefault(src, []).append(d)
            merged_docs: List[Document] = []
            for src, pages in grouped.items():
                if len(pages) == 1:
                    merged_docs.append(pages[0])
                    continue
                # Build concatenated text and record char spans for each page
                text_parts = []
                page_map = []  # list of {page, start, end}
                cursor = 0
                for p in sorted(pages, key=lambda x: x.metadata.get('page', 0)):
                    page_text = p.page_content.rstrip() + "\n\n"  # ensure boundary separation
                    start = cursor
                    cursor += len(page_text)
                    end = cursor
                    page_map.append({"page": p.metadata.get('page'), "start": start, "end": end})
                    text_parts.append(page_text)
                merged = Document(
                    page_content="".join(text_parts),
                    metadata={
                        **pages[0].metadata,
                        "page_map": page_map,
                    },
                )
                merged_docs.append(merged)
            return self._structured.split(merged_docs)
        return self._basic.split_documents(docs)
