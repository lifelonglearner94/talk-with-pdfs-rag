from pathlib import Path
from app.core import RAGConfig
from app.core.hashing import compute_index_hash
from app.core.ingestion import PDFIngestor
import tempfile
import shutil
import pytest
from pypdf import PdfWriter

def _write_simple_pdf(path: Path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open('wb') as f:
        writer.write(f)

@pytest.fixture()
def tmp_pdf_dir():
    d = Path(tempfile.mkdtemp())
    _write_simple_pdf(d / "a.pdf")
    _write_simple_pdf(d / "b.pdf")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_hash_changes_with_chunk_size(tmp_pdf_dir):
    cfg1 = RAGConfig(data_dir=tmp_pdf_dir, chunk_size=500)
    cfg2 = RAGConfig(data_dir=tmp_pdf_dir, chunk_size=800)
    pdfs = sorted(tmp_pdf_dir.glob('*.pdf'))
    h1 = compute_index_hash(pdfs, cfg1)
    h2 = compute_index_hash(pdfs, cfg2)
    assert h1 != h2


def test_ingestion_skips_zone_identifier(tmp_pdf_dir):
    # create artifact file with pattern
    _write_simple_pdf(tmp_pdf_dir / 'ignore:Zone.Identifier.pdf')
    ing = PDFIngestor(tmp_pdf_dir)
    docs = ing.ingest()
    names = {Path(d.metadata.get('source', '')).name for d in docs}
    assert 'ignore:Zone.Identifier.pdf' not in names
