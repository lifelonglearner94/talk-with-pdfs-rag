from __future__ import annotations
from pathlib import Path
from typing import Optional

# Simple citation key + formatting utilities (future: parse metadata)

def derive_citation_key(source_path: str | Path) -> str:
    name = Path(source_path).name
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    return name.replace(' ', '_')


def format_citation_inline(citation_key: str, year: Optional[str] = None) -> str:
    if year:
        return f"({citation_key} {year})"
    return f"({citation_key})"
