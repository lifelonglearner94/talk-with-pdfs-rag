from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from .preprocessing import preprocess_documents
from .config import RAGConfig
from .logging import logger

class PDFIngestor:
    def __init__(self, data_dir: Path, config: RAGConfig | None = None):
        self.data_dir = data_dir
        self.config = config

    def ingest(self) -> List[Document]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        pdfs = [p for p in self.data_dir.glob("*.pdf") if ':Zone.Identifier' not in p.name]
        if not pdfs:
            raise FileNotFoundError("No PDF files found in data directory")
        logger.info("Loading %d PDF files from %s", len(pdfs), self.data_dir)
        loader = PyPDFDirectoryLoader(str(self.data_dir))
        docs = loader.load()
        # Filter any Zone.Identifier artifacts in case loader still ingested them
        docs = [d for d in docs if ':Zone.Identifier' not in Path(d.metadata.get('source','')).name]
        if not docs:
            raise ValueError("No documents were loaded from PDFs")

        # Apply preprocessing if config is provided
        if self.config and self.config.remove_bibliography:
            docs = preprocess_documents(docs, remove_bib=True)

        return docs
