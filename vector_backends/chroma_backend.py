"""Chroma backend adapter (lightweight).

This adapter intentionally keeps dependencies optional. It attempts to import
chroma if available; otherwise raises a RuntimeError at runtime when used.
"""
from typing import Sequence, Mapping, Any, Optional

from .base import VectorBackend


class ChromaBackend(VectorBackend):
    def __init__(self, persist_directory: Optional[str] = None):
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as e:
            raise RuntimeError("chroma is not installed: " + str(e))

        self._persist_directory = persist_directory
        self._client = chromadb.Client(Settings(persist_directory=persist_directory))

    def create_index(self, index_name: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        # chroma creates collections on the fly when adding
        self._client.create_collection(name=index_name)

    def add_documents(self, index_name: str, docs: Sequence[Mapping[str, Any]]) -> None:
        collection = self._client.get_collection(index_name)
        ids = [d.get("id") for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        documents = [d.get("text", "") for d in docs]
        embeddings = [d.get("embedding") for d in docs]
        collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)

    def search(self, index_name: str, query_embedding: Sequence[float], top_k: int = 10) -> Sequence[Mapping[str, Any]]:
        collection = self._client.get_collection(index_name)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        # results: dict with ids, distances, documents, metadatas
        hits = []
        for i, _ in enumerate(results.get("ids", [[]])[0]):
            hit = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None,
                "document": results["documents"][0][i] if "documents" in results else None,
                "metadata": results["metadatas"][0][i] if "metadatas" in results else None,
            }
            hits.append(hit)
        return hits

    def delete_index(self, index_name: str) -> None:
        self._client.delete_collection(name=index_name)
