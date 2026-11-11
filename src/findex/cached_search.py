"""Cached Boolean query evaluation."""

import logging
from functools import lru_cache
from typing import Protocol

from .index_model import Index
from .query_parser import parse_query

logger = logging.getLogger(__name__)


class CacheInfo(Protocol):
    @property
    def hits(self) -> int: ...

    @property
    def misses(self) -> int: ...

    @property
    def maxsize(self) -> int | None: ...

    @property
    def currsize(self) -> int: ...


class CachedQuery(Protocol):
    def __call__(self, query: str) -> tuple[int, ...]: ...

    def cache_info(self) -> CacheInfo: ...


def make_cached_query(index: Index) -> CachedQuery:
    @lru_cache(maxsize=256)
    def query_to_doc_ids(query: str) -> tuple[int, ...]:
        normalized_query = query.strip().lower()
        result = parse_query(normalized_query).evaluate(index)
        return tuple(sorted(result))

    return query_to_doc_ids


def demo_cache(index: Index, query: str) -> None:
    cached_query = make_cached_query(index)
    logger.info("First query: %s", cached_query(query))
    logger.info("Cache after first query: %s", cached_query.cache_info())
    logger.info("Repeated query: %s", cached_query(query))
    logger.info("Cache after repeated query: %s", cached_query.cache_info())
