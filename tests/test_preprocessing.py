from src.preprocessing.mind_preprocess import normalize_text, parse_impressions


def test_normalize_text_basic():
    s = "The Quick, brown fox!"
    out = normalize_text(s)
    assert "quick" in out
    assert "," not in out


def test_parse_impressions():
    s = "N1-1 N2-0 N3-1"
    items = parse_impressions(s)
    assert isinstance(items, list)
    assert items[0] == ("N1", 1)
    assert items[1] == ("N2", 0)
