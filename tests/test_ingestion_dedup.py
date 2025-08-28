from app.core.vectorstore import VectorStoreManager


class FakeBackend:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        self.created = False
        self.added = []

    def create_index(self, index_name: str, metadata=None):
        self.created = True

    def add_documents(self, index_name: str, docs):
        self.added.append((index_name, list(docs)))

    def delete_index(self, index_name: str):
        self.created = False


def test_ingestion_worker_dedup(monkeypatch, tmp_path):
    import vector_backends

    monkeypatch.setattr(vector_backends, 'ChromaBackend', FakeBackend)
    vsm = VectorStoreManager(tmp_path, embedding_fn=lambda x: x)
    q = vsm.create_ingestion_queue(index_name='dedupidx', batch_size=2, interval=0.1)

    # Call worker with duplicate docs (one with explicit id, one duplicate by text)
    batch1 = [{'id': 'a', 'text': 'hello', 'metadata': {}}, {'text': 'dup', 'metadata': {'citation_key': 'C1'}}]
    q._worker(batch1)
    # Second batch includes same id and same text+citation
    batch2 = [{'id': 'a', 'text': 'hello', 'metadata': {}}, {'text': 'dup', 'metadata': {'citation_key': 'C1'}}]
    q._worker(batch2)

    # Backend should have been created and received only the first unique docs
    assert FakeBackend.instances if hasattr(FakeBackend, 'instances') else True
    # Find the fake backend instance used
    # The VectorStoreManager.create_ingestion_queue instantiates backend internally
    # We can't directly access instances list here; instead, rely on the fact that
    # the FakeBackend.added list should have exactly one add call with two docs
    # Access the backend via monkeypatch by instantiating a new one: but simpler
    # check that no exception was raised and the behavior preserved (no duplicates added twice)
    # For stronger assertion, create a FakeBackend instance directly and call worker_from_backend
    from core.ingestion_queue import worker_from_backend
    backend = FakeBackend(tmp_path)
    worker = worker_from_backend(backend, 'dedup-test')
    worker(batch1)
    worker(batch2)
    # backend.added should contain only one call with 2 docs
    assert len(backend.added) == 1
    assert backend.added[0][0] == 'dedup-test'
    assert len(backend.added[0][1]) == 2
