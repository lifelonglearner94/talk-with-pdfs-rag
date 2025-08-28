import asyncio

from core.ingestion_queue import IngestionQueue


async def _dummy_worker(batch):
    # collect sum of lengths for verification
    return len(batch)


def test_ingestion_queue_runs():
    q = IngestionQueue(worker=_dummy_worker, batch_size=2, interval=0.1)

    async def produce_and_consume():
        await q.start()
        await q.push({"id": 1})
        await q.push({"id": 2})
        # allow queue to process
        await asyncio.sleep(0.3)
        await q.stop()

    # Use asyncio.run so the test doesn't rely on pytest's event_loop fixture
    asyncio.run(produce_and_consume())
    assert q._stopped is True
