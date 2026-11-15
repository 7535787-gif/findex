"""Adapter between HTTP handlers and the existing findex search core."""

from __future__ import annotations

import json
from pathlib import Path

from findex.index import DocMeta
from findex.index_model import Index
from findex.ranking import (
    BM25,
    SearchResult as CoreSearchResult,
    TfIdf,
    search as ranked_search,
)
from findex.tokenize import tokenize


def run_search(
    index: Index, query: str, scorer_name: str, k: int
) -> list[CoreSearchResult]:
    terms = list(tokenize(query))
    if not terms:
        return []
    scorer = BM25() if scorer_name == "bm25" else TfIdf()
    return ranked_search(index, terms, scorer, top_k=k)


def read_document(document: DocMeta) -> str:
    """Read a source document, including one record from crawler JSONL."""
    if document.content_path and document.line_offset is not None:
        path = Path(document.content_path)
        with path.open("rb") as stream:
            stream.seek(document.line_offset)
            raw = stream.readline()
        record = json.loads(raw.decode("utf-8"))
        return str(record.get("text", ""))
    return Path(document.path).read_text(encoding="utf-8", errors="replace")
