from app.core.metadata_extraction import extract_basic_metadata

def test_extract_basic_metadata_year_fallback_none():
    meta = extract_basic_metadata('Smith_Scalable_Study.pdf')
    assert meta['year'] is None
