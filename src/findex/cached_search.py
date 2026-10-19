
from functools import lru_cache

from .index_model import Index
from .query_parser import parse_query


def make_cached_query(index: Index):
    @lru_cache(maxsize=256)
    def query_to_doc_ids(query: str) -> tuple[int, ...]:
        normalized_query = query.strip().lower()
        result = parse_query(normalized_query).evaluate(index)
        return tuple(sorted(result))

    return query_to_doc_ids


def demo_cache(index: Index, query: str) -> None:
    cached_query = make_cached_query(index)

    print("First query:", cached_query(query))
    print("Cache after first query:", cached_query.cache_info())

    print("Repeated query:", cached_query(query))
    print("Cache after repeated query:", cached_query.cache_info())
