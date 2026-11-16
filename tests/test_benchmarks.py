"""pytest-benchmark workloads for regression tracking in CI."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from findex.index import build_index
from findex.ranking import BM25, search
from findex.index_model import Index


def test_benchmark_search(benchmark, tiny_index: Index):
    result = benchmark(
        search, tiny_index, ["python", "search", "engine"], BM25(), 5
    )
    assert result


def test_benchmark_build_index(benchmark, tiny_corpus):
    postings, documents = benchmark(build_index, tiny_corpus)
    assert postings
    assert documents
