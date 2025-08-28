from app.core.rag_pipeline import RAGPipeline
from app.core.config import RAGConfig
from langchain_core.documents import Document

class DummyPipeline(RAGPipeline):
    def __init__(self):
        # minimal config with structure mode
        cfg = RAGConfig(chunking_mode="structure")
        super().__init__(cfg)


def test_deterministic_chunk_id_schema():
    # simulate already split structured docs (3 chunks two sections)
    docs = [
        Document(page_content="Intro text 1", metadata={"source": "Paper, 2024, Title.pdf", "section_index": 0}),
        Document(page_content="Intro text 2", metadata={"source": "Paper, 2024, Title.pdf", "section_index": 0}),
        Document(page_content="Methods text", metadata={"source": "Paper, 2024, Title.pdf", "section_index": 1}),
    ]
    pipe = DummyPipeline()
    enhanced = pipe._enhance_metadata(docs)
    chunk_ids = [d.metadata['chunk_id'] for d in enhanced]
    # Expect pattern citation_key:section_index:local_idx with increasing local_idx per section
    assert chunk_ids[0].endswith(":0:0")
    assert chunk_ids[1].endswith(":0:1")
    assert chunk_ids[2].endswith(":1:0")
    # All unique
    assert len(set(chunk_ids)) == len(chunk_ids)
