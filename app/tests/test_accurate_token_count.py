import types
import sys
from app.core.config import RAGConfig
from app.core.structured_splitter import StructuredPaperSplitter
from app.core.splitting import DocumentSplitter
from langchain_core.documents import Document


class DummyEncoding:
    def encode(self, text: str):
        # deterministic: one token per word
        return text.split()


def install_fake_tiktoken(monkeypatch):
    mod = types.ModuleType('tiktoken')
    def get_encoding(name):
        return DummyEncoding()
    mod.get_encoding = get_encoding  # type: ignore
    sys.modules['tiktoken'] = mod


def test_structured_splitter_uses_token_counter(monkeypatch):
    install_fake_tiktoken(monkeypatch)
    cfg = RAGConfig(accurate_token_count=True, chunking_mode='structure', chunk_size=100, chunk_overlap=0)
    splitter = DocumentSplitter(cfg.chunk_size, cfg.chunk_overlap, cfg)
    doc = Document(page_content="Intro\n\nIntroduction\nThis is a test paragraph with several words.", metadata={"source":"X.pdf","page":0})
    chunks = splitter.split([doc])
    # With dummy encoding (word count) tokens should equal number of words in chunk content roughly
    for c in chunks:
        tc = c.metadata.get('token_count')
        assert tc is not None and tc < len(c.page_content), "Expected word-based token count smaller than char length"
