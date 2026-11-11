from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import (
    DocMeta,
    Posting,
    and_merge,
    build_index,
    load_index,
    not_merge,
    or_merge,
    save_index,
)
from findex.index_model import Index
from findex.query_parser import And, Or, Term, parse_query
from findex.ranking import BM25, TfIdf, search
from findex.store import load_json, save_json
from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("КИЇВ Україна", ["київ", "україна"]),
        ("cafe\u0301", ["café"]),
        ("комп'ютер", ["комп'ютер"]),
        ("об’єкт", ["об’єкт"]),
        ("a-b", ["a", "b"]),
        ("123", []),
        ("", []),
        ("Привіт, світе!", ["привіт", "світе"]),
        ("naïve café", ["naïve", "café"]),
    ],
)
def test_tokenizer_parametrized(source: str, expected: list[str]) -> None:
    assert list(tokenize(source)) == expected


def test_fixture_builds_small_index(tiny_index: Index) -> None:
    assert tiny_index.num_docs == 3
    assert tiny_index.df("python") == 2


def test_index_words_are_present(tiny_index: Index) -> None:
    assert "python" in tiny_index
    assert "async" in tiny_index


def test_index_words_can_be_missing(tiny_index: Index) -> None:
    assert "nonexistent" not in tiny_index


def test_average_document_length_positive(tiny_index: Index) -> None:
    assert tiny_index.avg_doc_length > 0


def test_doc_length(tiny_index: Index) -> None:
    document = next(
        doc for doc in tiny_index.documents.values() if Path(doc.path).name == "one.txt"
    )
    assert tiny_index.doc_length(document.doc_id) == 4


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2, 3], [2, 3, 4], [2, 3]),
        ([], [1], []),
        ([1], [], []),
        ([1, 3, 5], [2, 4, 6], []),
        ([1, 2], [1, 2], [1, 2]),
    ],
)
def test_and_merge(left: list[int], right: list[int], expected: list[int]) -> None:
    assert and_merge(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2], [2, 3], [1, 2, 3]),
        ([], [1], [1]),
        ([1], [], [1]),
        ([], [], []),
        ([1, 3], [2, 4], [1, 2, 3, 4]),
    ],
)
def test_or_merge(left: list[int], right: list[int], expected: list[int]) -> None:
    assert or_merge(left, right) == expected


def test_not_merge() -> None:
    assert not_merge([0, 1, 2, 3], [1, 3]) == [0, 2]


def test_parser_term_tree() -> None:
    assert parse_query("python") == Term("python")


def test_parser_and_tree() -> None:
    assert parse_query("python AND async") == And(Term("python"), Term("async"))


def test_parser_or_tree() -> None:
    assert parse_query("python OR async") == Or(Term("python"), Term("async"))


def test_parser_precedence_tree() -> None:
    assert parse_query("python AND async OR java") == Or(
        And(Term("python"), Term("async")), Term("java")
    )


@pytest.mark.parametrize("query", ["", "()", "python AND", "(python", "python)"])
def test_parser_rejects_invalid_queries(query: str) -> None:
    with pytest.raises(ValueError):
        parse_query(query)


def test_tfidf_score_is_positive(tiny_index: Index) -> None:
    assert TfIdf().score("python", 1, tiny_index, 0) > 0


def test_bm25_score_is_positive(tiny_index: Index) -> None:
    assert BM25().score("python", 1, tiny_index, 0) > 0


def test_bm25_tf_saturation(tiny_index: Index) -> None:
    scorer = BM25()
    low = scorer.score("python", 1, tiny_index, 0)
    high = scorer.score("python", 20, tiny_index, 0)
    assert high > low
    assert high < low * 20


def test_ranked_search_returns_top_k(tiny_index: Index) -> None:
    assert len(search(tiny_index, ["python"], BM25(), top_k=1)) <= 1


def test_ranked_search_empty_query(tiny_index: Index) -> None:
    assert search(tiny_index, [], BM25()) == []


def test_pickle_save_load_roundtrip(tiny_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "test.bin"
    save_index(dict(tiny_index.items()), tiny_index.documents, path)
    postings, documents, _ = load_index(path)
    assert postings == {term: list(items) for term, items in tiny_index.items()}
    assert documents == tiny_index.documents


runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.output and "search" in result.output


def test_cli_index_command(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.bin"
    result = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_cli_search_table(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.bin"
    runner.invoke(app, ["index", str(tiny_corpus), "--out", str(output)])
    result = runner.invoke(app, ["search", str(output), "python"])
    assert result.exit_code == 0, result.output
    assert "Search results" in result.output


def test_cli_search_json_stdout(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.bin"
    runner.invoke(app, ["index", str(tiny_corpus), "--out", str(output)])
    result = runner.invoke(app, ["search", str(output), "python", "--json"])
    assert result.exit_code == 0, result.output

    payload = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert payload
    assert all("doc_id" in item for item in payload)


def test_cli_missing_index_has_nonzero_exit() -> None:
    result = runner.invoke(app, ["search", "missing.bin", "python"])
    assert result.exit_code != 0
    assert "Index file not found" in result.output


def test_cli_stats(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.bin"
    runner.invoke(app, ["index", str(tiny_corpus), "--out", str(output)])
    result = runner.invoke(app, ["stats", str(output)])
    assert result.exit_code == 0
    assert "Documents" in result.output


def test_json_save_load_roundtrip(tiny_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    save_json(dict(tiny_index.items()), tiny_index.documents, path)
    postings, documents, elapsed = load_json(path)
    assert elapsed >= 0
    assert documents == tiny_index.documents
    assert {
        term: [(p.doc_id, p.tf) for p in values] for term, values in postings.items()
    } == {
        term: [(p.doc_id, p.tf) for p in values] for term, values in tiny_index.items()
    }


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_has_higher_idf_than_common_term(scorer: TfIdf | BM25) -> None:
    index = Index(
        {
            "rare": [Posting(0, 1)],
            "common": [Posting(0, 1), Posting(1, 1), Posting(2, 1)],
        },
        {
            0: DocMeta(0, "a.txt", "a", 10),
            1: DocMeta(1, "b.txt", "b", 10),
            2: DocMeta(2, "c.txt", "c", 10),
        },
    )
    assert scorer.score("rare", 1, index, 0) > scorer.score("common", 1, index, 0)


def test_tfidf_logarithmic_tf_growth(tiny_index: Index) -> None:
    scorer = TfIdf()
    low = scorer.score("python", 1, tiny_index, 0)
    high = scorer.score("python", 20, tiny_index, 0)
    assert high > low
    assert high < low * 20


def test_bm25_prefers_shorter_document_for_equal_tf() -> None:
    index = Index(
        {"term": [Posting(0, 1), Posting(1, 1)]},
        {
            0: DocMeta(0, "short.txt", "short", 10),
            1: DocMeta(1, "long.txt", "long", 100),
        },
    )
    scorer = BM25()
    assert scorer.score("term", 1, index, 0) > scorer.score("term", 1, index, 1)


def test_build_index_limit(tiny_corpus: Path) -> None:
    _, documents = build_index(tiny_corpus, limit=1)
    assert len(documents) == 1


def test_build_index_without_positions(tiny_corpus: Path) -> None:
    postings, _ = build_index(tiny_corpus, include_positions=False)
    assert postings
    assert all(
        not posting.positions for items in postings.values() for posting in items
    )


def test_cli_index_positions_and_limit(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "limited.bin"
    result = runner.invoke(
        app,
        [
            "index",
            str(tiny_corpus),
            "--out",
            str(output),
            "--positions",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    postings, documents, _ = load_index(output)
    assert len(documents) == 1
    assert any(posting.positions for items in postings.values() for posting in items)


def test_cli_search_k_option(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli-k.bin"
    runner.invoke(app, ["index", str(tiny_corpus), "--out", str(output)])
    result = runner.invoke(app, ["search", str(output), "python", "--k", "1"])
    assert result.exit_code == 0, result.output
