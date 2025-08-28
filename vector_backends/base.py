from typing import Sequence, Mapping, Any, Optional


class VectorBackend:
    """Abstract vector backend interface.

    Implementations should provide create_index, add_documents, search, and delete methods.
    """

    def create_index(self, index_name: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        raise NotImplementedError()

    def add_documents(self, index_name: str, docs: Sequence[Mapping[str, Any]]) -> None:
        raise NotImplementedError()

    def search(self, index_name: str, query_embedding: Sequence[float], top_k: int = 10) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError()

    def delete_index(self, index_name: str) -> None:
        raise NotImplementedError()
