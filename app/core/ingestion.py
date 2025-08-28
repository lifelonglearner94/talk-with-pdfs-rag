from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

class PDFIngestor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def ingest(self) -> List[Document]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        pdfs = [p for p in self.data_dir.glob("*.pdf") if ':Zone.Identifier' not in p.name]
        if not pdfs:
            raise FileNotFoundError("No PDF files found in data directory")
        loader = PyPDFDirectoryLoader(str(self.data_dir))
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from PDFs")
        return docs
