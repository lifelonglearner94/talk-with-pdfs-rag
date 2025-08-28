from pathlib import Path
import json, tempfile
from app.core.vectorstore import VectorStoreManager
from app.core.config import RAGConfig
from app.core.hashing import compute_index_hash, INDEX_FORMAT_VERSION
from langchain_core.documents import Document


def test_needs_rebuild_on_version_mismatch(tmp_path: Path):
    # Create fake PDF file to feed into hash
    pdf = tmp_path / "Dummy, 2024, Sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    cfg = RAGConfig(data_dir=tmp_path, persist_dir=tmp_path / "vs")
    vs = VectorStoreManager(cfg.persist_dir, embedding_fn=lambda x: [[0.0]])  # embedding fn won't be used
    # Prepare docs (only metadata.source is used for hashing)
    docs = [Document(page_content="text", metadata={"source": str(pdf)})]
    # Precompute current hash
    current_hash = compute_index_hash([pdf], cfg)
    # Write older version state file with same hash but downgraded version number to force rebuild
    old_state = {"hash": current_hash, "version": INDEX_FORMAT_VERSION - 1}
    (cfg.persist_dir / "index_state.json").write_text(json.dumps(old_state))
    # Should request rebuild due to version mismatch
    assert vs.needs_rebuild(docs, cfg) is True
