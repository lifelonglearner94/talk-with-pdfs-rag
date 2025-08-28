from __future__ import annotations
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
