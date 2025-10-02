from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class ChunkMetadata:
    source_name: str
    chunk_id: str
    page: Optional[int] = None
    score: Optional[float] = None
    section: Optional[str] = None
    section_index: Optional[int] = None
    section_level: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: Optional[int] = None
    splitting_mode: Optional[str] = None

@dataclass
class RetrievalResult:
    text: str
    metadata: ChunkMetadata

@dataclass
class AnswerResult:
    answer: str
    sources: List[ChunkMetadata]
    raw_chunks: List[RetrievalResult]
