import math
from dataclasses import dataclass
from heapq import nlargest
from typing import Protocol

import numpy as np

from .index_model import Index
from .snippets import make_snippet
from .timing import timed
from .tokenize import tokenize


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    path: str
    snippet: str = ""
    match_type: str = "term"


class Scorer(Protocol):
    def score(self, term: str, tf: int, index: Index, doc_id: int) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, tf: int, index: Index, doc_id: int) -> float:
        if tf <= 0 or index.num_docs == 0:
            return 0.0

        df = index.df(term)
        if df == 0:
            return 0.0

        idf = math.log((index.num_docs + 1) / (df + 1)) + 1
        return (1 + math.log(tf)) * idf


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, tf: int, index: Index, doc_id: int) -> float:
        if tf <= 0 or index.num_docs == 0:
            return 0.0

        df = index.df(term)
        if df == 0:
            return 0.0

        idf = math.log(1 + (index.num_docs - df + 0.5) / (df + 0.5))
        doc_length = index.documents[doc_id].length
        avg_length = index.avg_doc_length

        if avg_length == 0:
            return 0.0

        length_norm = 1 - self.b + self.b * doc_length / avg_length
        tf_component = tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return idf * tf_component


@timed
def search(
    index: Index,
    query_terms: list[str],
    scorer: Scorer,
    top_k: int = 5,
    *,
    phrase_terms: list[str] | None = None,
    phrase_doc_ids: set[int] | None = None,
) -> list[SearchResult]:
    """Ранжує документи й додає snippets до результатів."""
    if top_k <= 0:
        return []

    normalized_terms = [
        term for query_term in query_terms for term in tokenize(query_term)
    ]

    scores: dict[int, float] = {}

    for term in normalized_terms:
        for posting in index.get(term, ()):
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + scorer.score(
                term, posting.tf, index, posting.doc_id
            )

    results: list[SearchResult] = []

    for doc_id, score in scores.items():
        document = index.documents[doc_id]
        snippet = make_snippet(
            document.content_path or document.path,
            normalized_terms,
            line_offset=document.line_offset,
            phrase_terms=phrase_terms,
        )

        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=document.title,
                path=document.path,
                snippet=snippet,
                match_type=(
                    "phrase"
                    if phrase_doc_ids is not None and doc_id in phrase_doc_ids
                    else ("partial" if phrase_terms is not None else "term")
                ),
            )
        )

    return nlargest(top_k, results)



@timed
def search_numpy(
    index: Index,
    query_terms: list[str],
    scorer: Scorer,
    top_k: int = 5,
    *,
    phrase_terms: list[str] | None = None,
    phrase_doc_ids: set[int] | None = None,
) -> list[SearchResult]:
    """Vectorized BM25/TF-IDF ranking over int32 posting arrays.

    Candidate scoring uses NumPy ufuncs and ``np.add.at``. ``argpartition``
    selects the top-k candidates; ties at the boundary are resolved by doc_id
    exactly like the original dataclass ordering used by ``heapq.nlargest``.
    """
    if top_k <= 0 or index.num_docs == 0:
        return []

    normalized_terms = [
        term for query_term in query_terms for term in tokenize(query_term)
    ]
    if not normalized_terms:
        return []

    candidate_posting_count = sum(
        stop - start
        for term in normalized_terms
        if (posting_range := index.numpy_postings.get(term)) is not None
        for start, stop in [posting_range]
    )
    if candidate_posting_count < 64:
        # Small posting lists are faster with the low-overhead Python baseline.
        # NumPy is selected for workloads large enough to amortize array setup.
        return search(
            index,
            normalized_terms,
            scorer,
            top_k,
            phrase_terms=phrase_terms,
            phrase_doc_ids=phrase_doc_ids,
        )

    scores = np.zeros(len(index.doc_lengths), dtype=np.float64)
    lengths = index.doc_lengths.astype(np.float64, copy=False)
    avg_length = index.avg_doc_length

    for term in normalized_terms:
        posting_range = index.numpy_postings.get(term)
        if posting_range is None:
            continue
        start, stop = posting_range
        doc_ids = index.posting_doc_ids[start:stop]
        frequencies = index.posting_tfs[start:stop]
        if doc_ids.size == 0:
            continue
        df = int(doc_ids.size)
        if isinstance(scorer, BM25):
            if avg_length <= 0:
                continue
            idf = np.log1p((index.num_docs - df + 0.5) / (df + 0.5))
            doc_lengths = lengths[doc_ids]
            norm = 1.0 - scorer.b + scorer.b * doc_lengths / avg_length
            contribution = idf * (
                frequencies.astype(np.float64, copy=False) * (scorer.k1 + 1.0)
            ) / (
                frequencies.astype(np.float64, copy=False) + scorer.k1 * norm
            )
        elif isinstance(scorer, TfIdf):
            idf = np.log((index.num_docs + 1.0) / (df + 1.0)) + 1.0
            contribution = (1.0 + np.log(frequencies.astype(np.float64))) * idf
        else:
            # Preserve extensibility for third-party Scorer implementations.
            contribution = np.fromiter(
                (
                    scorer.score(term, int(tf), index, int(doc_id))
                    for doc_id, tf in zip(doc_ids, frequencies, strict=True)
                ),
                dtype=np.float64,
                count=doc_ids.size,
            )
        scores[doc_ids] += contribution

    candidate_ids = np.flatnonzero(scores > 0.0)
    if candidate_ids.size == 0:
        return []

    count = min(top_k, int(candidate_ids.size))
    candidate_scores = scores[candidate_ids]
    if count < candidate_ids.size and candidate_ids.size > 64:
        partition = np.argpartition(candidate_scores, -count)[-count:]
        threshold = float(candidate_scores[partition].min())
        selected = candidate_ids[candidate_scores >= threshold]
    else:
        # For tiny posting lists, partition setup costs more than sorting the
        # handful of candidates. Large lists use argpartition as required.
        selected = candidate_ids

    ordered_ids = sorted(
        (int(doc_id) for doc_id in selected),
        key=lambda doc_id: (float(scores[doc_id]), doc_id),
        reverse=True,
    )[:top_k]

    results: list[SearchResult] = []
    for doc_id in ordered_ids:
        document = index.documents[doc_id]
        results.append(
            SearchResult(
                score=float(scores[doc_id]),
                doc_id=doc_id,
                title=document.title,
                path=document.path,
                snippet=make_snippet(
                    document.content_path or document.path,
                    normalized_terms,
                    line_offset=document.line_offset,
                    phrase_terms=phrase_terms,
                ),
                match_type=(
                    "phrase"
                    if phrase_doc_ids is not None and doc_id in phrase_doc_ids
                    else ("partial" if phrase_terms is not None else "term")
                ),
            )
        )
    return results
