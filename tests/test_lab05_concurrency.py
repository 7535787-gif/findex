from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import (
    DocMeta,
    Posting,
    build_index,
    build_partial,
    merge,
    merge_partials,
)
from findex.tokenize import tokenize


def _snapshot(index, documents):
    return (
        {
            term: [(posting.doc_id, posting.tf, posting.positions) for posting in postings]
            for term, postings in sorted(index.items())
        },
        {
            doc_id: (doc.path, doc.title, doc.length)
            for doc_id, doc in documents.items()
        },
    )


def _lab04_reference(corpus: Path):
    """Reproduce the Lab 4 serial build_index algorithm as a regression oracle."""
    index: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}

    for doc_id, document in enumerate(iter_documents(corpus)):
        tokens = list(tokenize(document.text))
        counts = Counter(tokens)
        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=str(document.path),
            title=document.path.stem,
            length=len(tokens),
        )
        positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
        for position, term in enumerate(tokens):
            positions_by_term[term].append(position)
        for term, frequency in counts.items():
            index[term].append(
                Posting(doc_id, frequency, tuple(positions_by_term[term]))
            )

    for postings in index.values():
        postings.sort(key=lambda posting: posting.doc_id)
    return dict(index), documents


def test_new_serial_build_is_identical_to_lab04_reference(tiny_corpus: Path) -> None:
    expected = _lab04_reference(tiny_corpus)
    paths = tuple(str(path) for path in tiny_corpus.rglob("*.txt"))
    actual = merge([build_partial(paths, 0)])
    assert _snapshot(*actual) == _snapshot(*expected)


def test_serial_partials_merge_matches_build_index(tiny_corpus: Path) -> None:
    paths = list(tiny_corpus.rglob("*.txt"))
    chunks = [
        tuple(str(path) for path in paths[:2]),
        tuple(str(path) for path in paths[2:]),
    ]
    partials = [
        build_partial(chunks[0], 0),
        build_partial(chunks[1], 2),
    ]
    actual = merge_partials(partials)
    expected = build_index(tiny_corpus)
    assert _snapshot(*actual) == _snapshot(*expected)


@pytest.mark.parametrize("backend", ["serial", "threads", "processes"])
def test_executor_backends_are_equivalent(tiny_corpus: Path, backend: str) -> None:
    expected = build_index(tiny_corpus, executor="serial")
    actual = build_index(tiny_corpus, workers=2, executor=backend)
    assert _snapshot(*actual) == _snapshot(*expected)


def test_threaded_build_is_deterministic_across_repeated_runs(
    tiny_corpus: Path,
) -> None:
    expected = _snapshot(*build_index(tiny_corpus, executor="serial"))
    for _ in range(10):
        actual = _snapshot(
            *build_index(tiny_corpus, workers=4, executor="threads")
        )
        assert actual == expected


def test_partial_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Worker cannot read"):
        build_partial((str(tmp_path / "missing.txt"),), 0)


def test_merge_rejects_duplicate_doc_ids(tiny_corpus: Path) -> None:
    paths = list(tiny_corpus.rglob("*.txt"))
    partial = build_partial((str(paths[0]),), 0)
    with pytest.raises(ValueError, match="Duplicate document IDs"):
        merge_partials([partial, partial])


def test_cli_accepts_parallel_options(tiny_corpus: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from findex.cli import app

    output = tmp_path / "parallel-index.bin"
    result = CliRunner().invoke(
        app,
        [
            "index",
            str(tiny_corpus),
            "--out",
            str(output),
            "--workers",
            "2",
            "--executor",
            "processes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
