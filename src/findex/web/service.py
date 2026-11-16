"""Adapter between HTTP handlers and the existing findex search core."""

from __future__ import annotations

import json
from pathlib import Path

from findex.index import DocMeta
from findex.index_model import Index
from findex.query_parser import Phrase
from findex.ranking import (
    BM25,
    TfIdf,
    search_numpy,
)
from findex.ranking import SearchResult as CoreSearchResult
from findex.semantic import DEFAULT_MODEL, hybrid_search, semantic_search
from findex.tokenize import tokenize


def run_search(
    index: Index,
    query: str,
    scorer_name: str,
    k: int,
    *,
    mode: str | None = None,
    embeddings_path: str | Path = "embeddings.npy",
    model_name: str = DEFAULT_MODEL,
) -> list[CoreSearchResult]:
    """Rank all term matches, promoting documents containing the full phrase.

    A multi-token query uses a two-tier result policy:
    1. documents containing the complete token phrase in order;
    2. documents containing one or more query terms, ranked by BM25/TF-IDF.
    """
    terms = list(tokenize(query))
    if not terms or k <= 0:
        return []

    selected_mode = mode or scorer_name
    phrase_doc_ids: set[int] = (
        Phrase(query).evaluate(index)
        if len(terms) > 1 and selected_mode in {"bm25", "tfidf"}
        else set()
    )
    if selected_mode == "semantic":
        results = semantic_search(
            index, query, embeddings_path, model_name=model_name, top_k=k
        )
    elif selected_mode == "hybrid":
        results = hybrid_search(
            index, query, embeddings_path, model_name=model_name, top_k=k
        )
    elif selected_mode in {"bm25", "tfidf"}:
        scorer = BM25() if selected_mode == "bm25" else TfIdf()
        results = search_numpy(
            index,
            terms,
            scorer,
            top_k=k,
            phrase_terms=terms if len(terms) > 1 else None,
            phrase_doc_ids=phrase_doc_ids,
        )
    else:
        raise ValueError(f"Unknown search mode: {selected_mode}")
    if selected_mode in {"bm25", "tfidf"}:
        results = sorted(
            results,
            key=lambda item: (item.match_type == "phrase", item.score),
            reverse=True,
        )
    return results[:k]


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
