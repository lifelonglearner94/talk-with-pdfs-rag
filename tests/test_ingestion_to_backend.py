import asyncio
from core.ingestion_queue import IngestionQueue, worker_from_backend


class FakeBackend:
    def __init__(self):
        self.created = False
        self.added_batches = []

    def create_index(self, index_name: str):
        self.created = True

    def add_documents(self, index_name: str, docs):
        self.added_batches.append(list(docs))


def test_worker_from_backend_creates_and_adds(tmp_path):
    backend = FakeBackend()
    worker = worker_from_backend(backend, "test_idx")

    # call worker twice with different batches
    docs1 = [{"id": "a", "text": "t1", "metadata": {}}]
    docs2 = [{"id": "b", "text": "t2", "metadata": {}}]

    worker(docs1)
    worker(docs2)

    assert backend.created is True
    assert len(backend.added_batches) == 2
    assert backend.added_batches[0][0]["id"] == "a"
    assert backend.added_batches[1][0]["id"] == "b"


async def _run_queue_and_push(queue: IngestionQueue):
    await queue.start()
    await queue.push({"id": "x", "text": "x", "metadata": {}})
    await asyncio.sleep(0.1)
    await queue.stop()


def test_ingestion_queue_uses_worker(monkeypatch):
    backend = FakeBackend()
    worker = worker_from_backend(backend, "qidx")

    q = IngestionQueue(worker=worker, batch_size=2, interval=0.05)

    # run the queue in the event loop
    asyncio.run(_run_queue_and_push(q))

    # queue should have invoked backend.create_index and added at least one batch
    assert backend.created is True
    assert len(backend.added_batches) >= 1
