from __future__ import annotations
"""Lightweight metadata extraction heuristics (Phase 1 stub).

Current goals:
- Extract year from filename (first 4-digit sequence between 1900-2100) if present.
- Derive title (filename portion after year comma) heuristically.
- Derive primary author (substring before first comma) for citation key fallback.

Future expansion (planned):
- Parse first page text for title (largest font), authors list, DOI detection.
- Confidence scoring & fallback logic.
"""
from pathlib import Path
import re
from typing import Dict, Any

YEAR_RE = re.compile(r"(19|20|21)\d{2}")


def extract_basic_metadata(source_path: str | Path) -> Dict[str, Any]:
    path = Path(source_path)
    name = path.name
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    # Pattern assumption: Author, YEAR, Title of Paper
    parts = [p.strip() for p in name.split(',')]
    year = None
    for p in parts:
        m = YEAR_RE.search(p)
        if m:
            year = m.group(0)
            break
    title = None
    authors = []
    if parts:
        # first part is likely primary author surname
        primary_author = parts[0]
        if YEAR_RE.fullmatch(primary_author):  # guard mis-parse
            primary_author = None
        else:
            authors.append(primary_author)
    # Title assumption: portion after year token
    if year and year in name:
        # find segment after year comma
        after_year = name.split(year, 1)[1]
        after_year = after_year.lstrip(',').strip()
        title = after_year or None
    return {
        'year': year,
        'title': title,
        'authors': authors,
    }
