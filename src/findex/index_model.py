
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TypeVar

from .index import DocMeta, Posting, load_index

T = TypeVar("T")


class Index(Mapping[str, tuple[Posting, ...]]):
    """Об'єктна модель інвертованого індексу."""

    def __init__(
        self,
        postings: Mapping[str, list[Posting] | tuple[Posting, ...]],
        documents: Mapping[int, DocMeta],
    ) -> None:
        self._postings = {
            term: tuple(items)
            for term, items in postings.items()
        }
        self.documents = dict(documents)

    def __getitem__(self, term: str) -> tuple[Posting, ...]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self)}, "
            f"documents={self.num_docs})"
        )

    @property
    def num_docs(self) -> int:
        return len(self.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.documents:
            return 0.0
        return (
            sum(doc.length for doc in self.documents.values())
            / self.num_docs
        )

    def df(self, term: str) -> int:
        """Кількість документів, що містять термін."""
        return len(self._postings.get(term, ()))
    
def doc_length(self, doc_id: int) -> int:
    """Повертає кількість термінів у документі."""
    return self.documents[doc_id].length


@contextmanager
def open_index(path: str | Path):
    """Завантажує індекс і звільняє посилання після використання."""
    index_data, documents, _ = load_index(Path(path))
    index = Index(index_data, documents)
    try:
        yield index
    finally:
        del index
        del index_data
        del documents
