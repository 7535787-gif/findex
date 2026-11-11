from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from findex.index import (
    DocMeta,
    Posting,
    and_merge,
    build_index,
    load_index,
    or_merge,
    save_index,
)


@pytest.mark.slow
@given(
    st.lists(
        st.text(alphabet="abc xyz", min_size=1, max_size=20),
        min_size=0,
        max_size=8,
    )
)
def test_hypothesis_built_postings_are_sorted(documents: list[str]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        corpus = Path(directory) / "corpus"
        corpus.mkdir()
        for number, content in enumerate(documents):
            (corpus / f"{number}.txt").write_text(content, encoding="utf-8")
        index, _ = build_index(corpus)
        assert all(
            [posting.doc_id for posting in postings]
            == sorted(posting.doc_id for posting in postings)
            for postings in index.values()
        )


@given(
    st.sets(st.integers(min_value=0, max_value=100)),
    st.sets(st.integers(min_value=0, max_value=100)),
)
def test_hypothesis_and_merge_matches_sets(a: set[int], b: set[int]) -> None:
    assert and_merge(sorted(a), sorted(b)) == sorted(a & b)


@given(
    st.sets(st.integers(min_value=0, max_value=100)),
    st.sets(st.integers(min_value=0, max_value=100)),
)
def test_hypothesis_or_merge_matches_sets(a: set[int], b: set[int]) -> None:
    assert or_merge(sorted(a), sorted(b)) == sorted(a | b)


@pytest.mark.slow
@given(
    st.dictionaries(
        st.text(alphabet="abc", min_size=1, max_size=5),
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=20),
                st.integers(min_value=1, max_value=5),
            ),
            max_size=10,
        ),
        max_size=8,
    )
)
def test_hypothesis_save_load_roundtrip(
    data: dict[str, list[tuple[int, int]]],
) -> None:
    normalized = {
        term: [Posting(doc_id, tf) for doc_id, tf in sorted(set(items))]
        for term, items in data.items()
    }
    doc_ids = sorted(
        {posting.doc_id for items in normalized.values() for posting in items}
    )
    docs = {
        doc_id: DocMeta(doc_id, f"{doc_id}.txt", str(doc_id), 1) for doc_id in doc_ids
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.bin"
        save_index(normalized, docs, path)
        restored, restored_docs, _ = load_index(path)
    assert restored == normalized
    assert restored_docs == docs
