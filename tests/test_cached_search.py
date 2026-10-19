
from findex.cached_search import make_cached_query
from findex.index import DocMeta, Posting
from findex.index_model import Index


def test_repeated_query_uses_cache():
    index = Index(
        {"python": [Posting(0, 1, (0,))]},
        {0: DocMeta(0, "a.txt", "a", 1)},
    )

    cached_query = make_cached_query(index)

    first = cached_query("python")
    second = cached_query("python")

    assert first == (0,)
    assert second == (0,)
    assert cached_query.cache_info().hits == 1
    assert cached_query.cache_info().misses == 1
