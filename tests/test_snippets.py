
from findex.snippets import make_snippet


def test_snippet_highlights_query_term(tmp_path):
    document = tmp_path / "sample.txt"
    document.write_text(
        "This is an example document about Python programming.",
        encoding="utf-8",
    )

    snippet = make_snippet(document, ["python"])

    assert "**Python**" in snippet


def test_snippet_returns_unavailable_for_missing_file(tmp_path):
    snippet = make_snippet(tmp_path / "missing.txt", ["python"])

    assert snippet == "[Document text unavailable]"
