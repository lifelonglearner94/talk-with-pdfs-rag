#!/usr/bin/env python3
"""
Simple test script to verify bibliography removal is working with actual PDFs.
"""
from pathlib import Path
from app.core import RAGConfig, RAGPipeline
from app.core.ingestion import PDFIngestor
from app.core.preprocessing import detect_bibliography_start, remove_bibliography

def test_single_pdf(pdf_path: Path):
    """Test bibliography detection on a single PDF."""
    print(f"\n{'='*70}")
    print(f"Testing: {pdf_path.name}")
    print('='*70)

    # Load the PDF
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    # Combine all pages
    full_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"Total characters: {len(full_text)}")

    # Detect bibliography
    bib_pos = detect_bibliography_start(full_text)

    if bib_pos is None:
        print("❌ No bibliography section detected")
        return False
    else:
        print(f"✅ Bibliography detected at position {bib_pos}")

        # Show context around detection
        context_start = max(0, bib_pos - 100)
        context_end = min(len(full_text), bib_pos + 200)
        context = full_text[context_start:context_end]
        print(f"\nContext around detection:")
        print("-" * 70)
        print(context)
        print("-" * 70)

        # Remove bibliography
        cleaned, found = remove_bibliography(full_text)
        print(f"\nOriginal length: {len(full_text)}")
        print(f"Cleaned length: {len(cleaned)}")
        print(f"Removed: {len(full_text) - len(cleaned)} characters ({(len(full_text) - len(cleaned)) / len(full_text) * 100:.1f}%)")

        return True


def test_with_pipeline():
    """Test the full pipeline with bibliography removal enabled/disabled."""
    print("\n" + "="*70)
    print("Testing RAG Pipeline Integration")
    print("="*70)

    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ directory not found")
        return

    pdfs = list(data_dir.glob("*.pdf"))[:3]  # Test first 3 PDFs
    if not pdfs:
        print("❌ No PDFs found in data/")
        return

    print(f"Found {len(pdfs)} PDFs to test")

    # Test with bibliography removal enabled
    print("\n--- Testing with bibliography removal ENABLED ---")
    config = RAGConfig(remove_bibliography=True)
    ingestor = PDFIngestor(data_dir, config)
    docs_with_removal = ingestor.ingest()

    removed_count = sum(1 for doc in docs_with_removal if doc.metadata.get('bibliography_removed', False))
    print(f"✅ Processed {len(docs_with_removal)} documents")
    print(f"✅ Bibliography removed from {removed_count} documents")

    # Test with bibliography removal disabled
    print("\n--- Testing with bibliography removal DISABLED ---")
    config = RAGConfig(remove_bibliography=False)
    ingestor = PDFIngestor(data_dir, config)
    docs_without_removal = ingestor.ingest()

    print(f"✅ Processed {len(docs_without_removal)} documents")
    print(f"✅ No bibliography removal (as expected)")

    # Compare lengths
    if len(docs_with_removal) == len(docs_without_removal):
        total_removed = sum(
            len(doc_without.page_content) - len(doc_with.page_content)
            for doc_with, doc_without in zip(docs_with_removal, docs_without_removal)
        )
        print(f"\n📊 Total characters removed: {total_removed}")


def main():
    """Main test function."""
    print("Bibliography Removal Test Script")
    print("="*70)

    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ directory not found")
        print("Please create a data/ folder with some PDF files to test")
        return

    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDFs found in data/")
        return

    print(f"Found {len(pdfs)} PDF files in data/")

    # Test first few PDFs individually
    print("\n" + "="*70)
    print("Testing Individual PDFs")
    print("="*70)

    detected_count = 0
    for pdf in pdfs[:5]:  # Test first 5
        try:
            if test_single_pdf(pdf):
                detected_count += 1
        except Exception as e:
            print(f"❌ Error processing {pdf.name}: {e}")

    print(f"\n{'='*70}")
    print(f"Summary: Bibliography detected in {detected_count}/{min(5, len(pdfs))} PDFs")
    print('='*70)

    # Test pipeline integration
    try:
        test_with_pipeline()
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
