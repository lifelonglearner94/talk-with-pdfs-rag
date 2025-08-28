from app.core.metadata_extraction import extract_basic_metadata

def test_extract_basic_metadata_filename_pattern():
    meta = extract_basic_metadata('Smith, 2023, Interesting Study on Kubernetes.pdf')
    assert meta['year'] == '2023'
    assert 'Smith' in meta['authors'][0]
    assert 'Interesting Study' in (meta['title'] or '')
