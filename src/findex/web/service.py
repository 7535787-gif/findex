"""Adapter between HTTP handlers and the existing findex search core."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from findex.index import DocMeta
from findex.index_model import Index
from findex.query_parser import Phrase
from findex.ranking import (
    BM25,
    TfIdf,
)
from findex.ranking import (
    SearchResult as CoreSearchResult,
)
from findex.ranking import (
    search as ranked_search,
)
from findex.tokenize import tokenize


def run_search(
    index: Index, query: str, scorer_name: str, k: int
) -> list[CoreSearchResult]:
    """Rank all term matches, promoting documents containing the full phrase.

    A multi-token query uses a two-tier result policy:
    1. documents containing the complete token phrase in order;
    2. documents containing one or more query terms, ranked by BM25/TF-IDF.
    """
    terms = list(tokenize(query))
    if not terms or k <= 0:
        return []

    scorer = BM25() if scorer_name == "bm25" else TfIdf()
    phrase_doc_ids = Phrase(query).evaluate(index) if len(terms) > 1 else set()

    results = ranked_search(
        index,
        terms,
        scorer,
        top_k=index.num_docs,
        phrase_terms=terms if len(terms) > 1 else None,
        phrase_doc_ids=phrase_doc_ids,
    )
    if len(terms) == 1:
        results = [replace(item, match_type="term") for item in results]

    return sorted(
        results,
        key=lambda item: (item.match_type == "phrase", item.score),
        reverse=True,
    )[:k]


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
