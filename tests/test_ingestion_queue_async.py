import asyncio
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
        self.added.append((index_name, list(docs)))

    def delete_index(self, index_name: str):
        self.created = False


def test_ingestion_queue_processes_batches(monkeypatch, tmp_path):
    import vector_backends

    monkeypatch.setattr(vector_backends, 'ChromaBackend', FakeChromaBackend)

    vsm = VectorStoreManager(tmp_path, embedding_fn=lambda x: x)
    q = vsm.create_ingestion_queue(index_name='asyncidx', batch_size=2, interval=0.05)

    async def run_queue():
        await q.start()
        # Push two items to trigger a batch (batch_size=2)
        await q.push({'id': 'a', 'text': 'hello', 'metadata': {}})
        await q.push({'id': 'b', 'text': 'world', 'metadata': {}})
        # give the queue a moment to process
        await asyncio.sleep(0.2)
        await q.stop()

    asyncio.run(run_queue())

    assert FakeChromaBackend.instances, "Fake backend was not created"
    backend = FakeChromaBackend.instances[-1]
    assert backend.created is True
    # Should have received one batched call with 2 documents
    assert backend.added, "No batches were sent to backend"
    # find any batch that contains both ids
    found = any(len(batch[1]) >= 2 for batch in backend.added)
    assert found, f"Expected at least one batch of size>=2, got: {backend.added}"
