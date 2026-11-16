from __future__ import annotations

import pickle

import numpy as np

from findex.index import DocMeta, Posting, load_index, save_index
from findex.index_model import Index
from findex.ranking import BM25, TfIdf, search, search_numpy


def make_index(tmp_path):
    docs = {}
    postings = {
        "python": [Posting(0, 2), Posting(1, 1), Posting(3, 5)],
        "search": [Posting(0, 1), Posting(2, 2), Posting(3, 1)],
        "engine": [Posting(1, 3), Posting(2, 1)],
        "rare": [Posting(3, 1)],
    }
    for doc_id, length in enumerate([10, 20, 15, 40]):
        path = tmp_path / f"{doc_id}.txt"
        path.write_text(
            ("python search engine " * (length // 3 + 1)), encoding="utf-8"
        )
        docs[doc_id] = DocMeta(doc_id, str(path), f"doc-{doc_id}", length)
    return Index(postings, docs)


def test_numpy_arrays_are_int32(tmp_path):
    index = make_index(tmp_path)
    start, stop = index.numpy_postings["python"]
    assert index.posting_doc_ids[start:stop].dtype == np.int32
    assert index.posting_tfs[start:stop].dtype == np.int32
    assert index.doc_lengths.dtype == np.int32


def test_numpy_ranking_matches_lab3_order_and_scores(tmp_path):
    index = make_index(tmp_path)
    for scorer in (BM25(), TfIdf()):
        for terms in (["python"], ["rare"], ["python", "search", "engine"]):
            expected = search(index, terms, scorer, top_k=4)
            actual = search_numpy(index, terms, scorer, top_k=4)
            assert [item.doc_id for item in actual] == [
                item.doc_id for item in expected
            ]
            assert np.allclose(
                [item.score for item in actual],
                [item.score for item in expected],
                rtol=1e-14,
                atol=1e-14,
            )


def test_numpy_top_k_tie_break_matches_python(tmp_path):
    index = make_index(tmp_path)
    for scorer in (BM25(), TfIdf()):
        expected = search(index, ["search"], scorer, top_k=2)
        actual = search_numpy(index, ["search"], scorer, top_k=2)
        assert [item.doc_id for item in actual] == [
            item.doc_id for item in expected
        ]


def test_index_persists_compact_int32_postings(tmp_path):
    index = make_index(tmp_path)
    path = tmp_path / "compact.bin"
    save_index(
        {term: list(items) for term, items in index.items()},
        index.documents,
        path,
    )
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["format_version"] == 2
    assert payload["index"]["doc_ids"].dtype == np.int32
    assert payload["index"]["tfs"].dtype == np.int32
    assert payload["index"]["offsets"]["python"]
    assert payload["doc_lengths"].dtype == np.int32
    restored, documents, _ = load_index(path)
    restored_index = Index(restored, documents)
    assert restored_index.doc_lengths.dtype == np.int32
    assert search_numpy(restored_index, ["python"], BM25(), 4)


def test_old_index_format_remains_readable(tmp_path):
    index = make_index(tmp_path)
    old = {
        "index": {
            term: [(p.doc_id, p.tf, p.positions) for p in items]
            for term, items in index.items()
        },
        "documents": {
            doc_id: (d.path, d.title, d.length, d.content_path, d.line_offset)
            for doc_id, d in index.documents.items()
        },
    }
    path = tmp_path / "legacy.bin"
    with path.open("wb") as stream:
        pickle.dump(old, stream)
    postings, documents, _ = load_index(path)
    assert len(Index(postings, documents)) == len(index)



def test_vectorized_path_matches_reference_for_long_posting_lists(tmp_path):
    path = tmp_path / "shared.txt"
    path.write_text("python search", encoding="utf-8")
    documents = {
        doc_id: DocMeta(doc_id, str(path), f"doc-{doc_id}", 10 + doc_id % 7)
        for doc_id in range(256)
    }
    postings = {
        "python": [Posting(doc_id, 1 + doc_id % 4) for doc_id in range(256)],
        "search": [Posting(doc_id, 1) for doc_id in range(0, 256, 2)],
    }
    index = Index(postings, documents)
    for scorer in (BM25(), TfIdf()):
        expected = search(index, ["python", "search"], scorer, top_k=25)
        actual = search_numpy(index, ["python", "search"], scorer, top_k=25)
        assert [item.doc_id for item in actual] == [
            item.doc_id for item in expected
        ]
        assert np.allclose(
            [item.score for item in actual],
            [item.score for item in expected],
            rtol=1e-14,
            atol=1e-14,
        )
