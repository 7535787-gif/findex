from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np

from .index import DocMeta, Posting, load_index


class Index(Mapping[str, tuple[Posting, ...]]):
    """Об'єктна модель інвертованого індексу."""

    def __init__(
        self,
        postings: Mapping[str, list[Posting] | tuple[Posting, ...]],
        documents: Mapping[int, DocMeta],
    ) -> None:
        self._postings = {term: tuple(items) for term, items in postings.items()}
        self.documents = dict(documents)
        # Compact numeric view used by the vectorized ranking path. The original
        # Posting objects remain available for phrase queries and compatibility.
        total_postings = sum(len(items) for items in self._postings.values())
        self.posting_doc_ids = np.empty(total_postings, dtype=np.int32)
        self.posting_tfs = np.empty(total_postings, dtype=np.int32)
        self.numpy_postings: dict[str, tuple[int, int]] = {}
        cursor = 0
        for term, items in self._postings.items():
            stop = cursor + len(items)
            self.numpy_postings[term] = (cursor, stop)
            if items:
                self.posting_doc_ids[cursor:stop] = [item.doc_id for item in items]
                self.posting_tfs[cursor:stop] = [item.tf for item in items]
            cursor = stop
        max_doc_id = max(self.documents, default=-1)
        self.doc_lengths = np.zeros(max_doc_id + 1, dtype=np.int32)
        for doc_id, document in self.documents.items():
            if doc_id < 0:
                raise ValueError("Document IDs must be non-negative")
            self.doc_lengths[doc_id] = document.length

    def __getitem__(self, term: str) -> tuple[Posting, ...]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, documents={self.num_docs})"

    @property
    def num_docs(self) -> int:
        return len(self.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.documents:
            return 0.0
        return sum(doc.length for doc in self.documents.values()) / self.num_docs

    def df(self, term: str) -> int:
        """Кількість документів, що містять термін."""
        return len(self._postings.get(term, ()))

    def doc_length(self, doc_id: int) -> int:
        """Повертає кількість термінів у документі."""
        return self.documents[doc_id].length


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    """Завантажує індекс і звільняє посилання після використання."""
    index_data, documents, _ = load_index(Path(path))
    index = Index(index_data, documents)
    try:
        yield index
    finally:
        del index
        del index_data
        del documents
