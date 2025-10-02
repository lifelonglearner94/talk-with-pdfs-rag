from app.core import RAGPipeline, RAGConfig
import os
import pytest
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


REQUIRES_DATA = not Path('data').exists() or not list(Path('data').glob('*.pdf'))
MISSING_API_KEY = os.getenv('GOOGLE_API_KEY') in (None, '')

@pytest.mark.skipif(REQUIRES_DATA, reason='No PDFs available for test')
@pytest.mark.skipif(MISSING_API_KEY, reason='GOOGLE_API_KEY not set')
def test_pipeline_basic():
    cfg = RAGConfig()
    pipe = RAGPipeline(cfg)
    pipe.ensure_index()
    assert pipe._retriever is not None

@pytest.mark.skipif(REQUIRES_DATA, reason='No PDFs available for test')
@pytest.mark.skipif(MISSING_API_KEY, reason='GOOGLE_API_KEY not set')
def test_answer_sources():
    cfg = RAGConfig(top_k=2)
    pipe = RAGPipeline(cfg)
    res = pipe.answer('Kurze Zusammenfassung?')
    assert res.sources
