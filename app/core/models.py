from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class ChunkMetadata:
    source_name: str
    chunk_id: str
    page: Optional[int] = None
    score: Optional[float] = None

@dataclass
class RetrievalResult:
    text: str
    metadata: ChunkMetadata

@dataclass
class AnswerResult:
    answer: str
    sources: List[ChunkMetadata]
    raw_chunks: List[RetrievalResult]
