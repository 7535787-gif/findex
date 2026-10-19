
import math
from dataclasses import dataclass
from heapq import nlargest
from typing import Protocol

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


class Scorer(Protocol):
    def score(self, term: str, tf: int, index: Index, doc_id: int) -> float:
        ...


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

        idf = math.log(
            1 + (index.num_docs - df + 0.5) / (df + 0.5)
        )
        doc_length = index.documents[doc_id].length
        avg_length = index.avg_doc_length

        if avg_length == 0:
            return 0.0

        length_norm = 1 - self.b + self.b * doc_length / avg_length
        tf_component = (
            tf * (self.k1 + 1)
            / (tf + self.k1 * length_norm)
        )
        return idf * tf_component

@timed
def search(
    index: Index,
    query_terms: list[str],
    scorer: Scorer,
    top_k: int = 5,
) -> list[SearchResult]:
    """Ранжує документи й додає snippets до результатів."""
    if top_k <= 0:
        return []

    normalized_terms = [
        term
        for query_term in query_terms
        for term in tokenize(query_term)
    ]

    scores: dict[int, float] = {}

    for term in normalized_terms:
        for posting in index._postings.get(term, ()):
            scores[posting.doc_id] = (
                scores.get(posting.doc_id, 0.0)
                + scorer.score(
                    term, posting.tf, index, posting.doc_id
                )
            )

    results = []

    for doc_id, score in scores.items():
        document = index.documents[doc_id]
        snippet = make_snippet(document.path, normalized_terms)

        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=document.title,
                path=document.path,
                snippet=snippet,
            )
        )

    return nlargest(top_k, results)
