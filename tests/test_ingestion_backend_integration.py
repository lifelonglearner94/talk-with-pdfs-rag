from pathlib import Path
from app.core.vectorstore import VectorStoreManager


class FakeChromaBackend:
    instances = []

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        self.created = False
        self.added = []
        FakeChromaBackend.instances.append(self)

    def create_index(self, index_name: str, metadata=None):
        self.created = True

    def add_documents(self, index_name: str, docs):
        # record docs for assertion
        self.added.append((index_name, list(docs)))

    def delete_index(self, index_name: str):
        self.created = False


def test_create_ingestion_queue(monkeypatch, tmp_path):
    # Monkeypatch the vector_backends.ChromaBackend to our fake
    import vector_backends

    monkeypatch.setattr(vector_backends, 'ChromaBackend', FakeChromaBackend)

    # Create manager and request an ingestion queue
    vsm = VectorStoreManager(tmp_path, embedding_fn=lambda x: x)
    q = vsm.create_ingestion_queue(index_name='testidx', batch_size=2, interval=0.1)

    # The queue stores the worker callable; call it with a small batch synchronously
    batch = [{'id': 'd1', 'text': 'hello', 'metadata': {}}]
    # call the underlying worker
    q._worker(batch)

    # Ensure our fake backend instance was created and received the batch
    assert FakeChromaBackend.instances, "Fake backend was not instantiated"
    backend = FakeChromaBackend.instances[-1]
    assert backend.created is True
    assert backend.added, "Backend did not receive added documents"
    assert backend.added[0][0] == 'testidx'
    assert backend.added[0][1][0]['id'] == 'd1'
