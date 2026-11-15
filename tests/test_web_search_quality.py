from findex.index import DocMeta, Posting
from findex.index_model import Index
from findex.web.service import run_search


def test_exact_phrase_results_are_promoted_and_labeled(tmp_path):
    exact = tmp_path / "exact.txt"
    partial = tmp_path / "partial.txt"
    exact.write_text("The quick brown fox jumps.", encoding="utf-8")
    partial.write_text("A quick animal and a brown coat.", encoding="utf-8")

    index = Index(
        {
            "quick": [Posting(1, 1, (1,)), Posting(2, 1, (1,))],
            "brown": [Posting(1, 1, (2,)), Posting(2, 1, (5,))],
            "fox": [Posting(1, 1, (3,))],
        },
        {
            1: DocMeta(1, str(exact), "Exact", 5),
            2: DocMeta(2, str(partial), "Partial", 6),
        },
    )

    results = run_search(index, "quick brown fox", "bm25", 10)

    assert [item.doc_id for item in results] == [1, 2]
    assert results[0].match_type == "phrase"
    assert results[1].match_type == "partial"
    assert "**quick brown fox**" in results[0].snippet


def test_single_term_search_returns_all_matching_documents(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Python is useful.", encoding="utf-8")
    second.write_text("I also use Python.", encoding="utf-8")
    index = Index(
        {"python": [Posting(1, 1, (0,)), Posting(2, 1, (3,))]},
        {
            1: DocMeta(1, str(first), "First", 3),
            2: DocMeta(2, str(second), "Second", 4),
        },
    )

    results = run_search(index, "python", "bm25", 10)

    assert {item.doc_id for item in results} == {1, 2}
    assert all(item.match_type == "term" for item in results)
    assert all("**Python**" in item.snippet for item in results)


def test_multiword_query_keeps_partial_matches_after_phrase_matches(tmp_path):
    exact = tmp_path / "exact.txt"
    partial = tmp_path / "partial.txt"
    exact.write_text("secure network access", encoding="utf-8")
    partial.write_text("network tools provide secure features", encoding="utf-8")
    index = Index(
        {
            "secure": [Posting(1, 1, (0,)), Posting(2, 1, (4,))],
            "network": [Posting(1, 1, (1,)), Posting(2, 1, (0,))],
            "access": [Posting(1, 1, (2,))],
        },
        {
            1: DocMeta(1, str(exact), "Exact", 3),
            2: DocMeta(2, str(partial), "Partial", 5),
        },
    )

    results = run_search(index, "secure network access", "tfidf", 10)

    assert results[0].doc_id == 1
    assert results[0].match_type == "phrase"
    assert results[1].doc_id == 2
    assert results[1].match_type == "partial"



def test_api_and_ui_expose_phrase_first_results(tmp_path):
    from fastapi.testclient import TestClient

    from findex.web.app import create_app, optional_index
    from findex.web.config import Settings
    from findex.web.dependencies import get_index

    exact = tmp_path / "exact.txt"
    partial = tmp_path / "partial.txt"
    exact.write_text("secure network access", encoding="utf-8")
    partial.write_text("network tools provide secure features", encoding="utf-8")
    index = Index(
        {
            "secure": [Posting(1, 1, (0,)), Posting(2, 1, (4,))],
            "network": [Posting(1, 1, (1,)), Posting(2, 1, (0,))],
            "access": [Posting(1, 1, (2,))],
        },
        {
            1: DocMeta(1, str(exact), "Exact", 3),
            2: DocMeta(2, str(partial), "Partial", 5),
        },
    )
    app = create_app(Settings(index_path=None), load_on_startup=False)
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[optional_index] = lambda: index

    with TestClient(app) as client:
        response = client.get("/search", params={"q": "secure network access"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["results"][0]["match_type"] == "phrase"
        assert payload["results"][1]["match_type"] == "partial"

        page = client.get("/", params={"q": "secure network access"})
        assert page.status_code == 200
        assert page.text.index("ТОЧНА ФРАЗА") < page.text.index("ЧАСТКОВИЙ ЗБІГ")



def test_phrase_snippet_handles_latex_separators(tmp_path):
    from findex.snippets import make_snippet

    source = tmp_path / "formula.txt"
    source.write_text(r"Formula {\displaystyle \r\ggr_{s}} follows.", encoding="utf-8")

    snippet = make_snippet(
        source,
        ["displaystyle", "r", "ggr", "s"],
        phrase_terms=["displaystyle", "r", "ggr", "s"],
    )

    assert "**displaystyle" in snippet and r"\r\ggr_{s**" in snippet
