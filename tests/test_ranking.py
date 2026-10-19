
from findex.index import DocMeta, Posting
from findex.index_model import Index
from findex.ranking import BM25, TfIdf, search


def make_index():
    postings = {
        "python": [Posting(0, 1), Posting(1, 20)],
        "rare": [Posting(0, 1)],
    }
    documents = {
        0: DocMeta(0, "short.txt", "short", 10),
        1: DocMeta(1, "long.txt", "long", 1000),
    }
    return Index(postings, documents)


def test_scorers_return_ranked_results():
    index = make_index()

    for scorer in (TfIdf(), BM25()):
        results = search(index, ["rare", "python"], scorer, top_k=2)

        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert all(result.title for result in results)


def test_bm25_tf_saturation():
    index = make_index()
    scorer = BM25()

    one = scorer.score("python", 1, index, 0)
    twenty = scorer.score("python", 20, index, 1)

    assert twenty > one
    assert twenty < one * 20


def test_top_k_and_empty_query():
    index = make_index()

    assert len(search(index, ["python"], BM25(), top_k=1)) == 1
    assert search(index, ["unknown"], BM25()) == []
    assert search(index, [], BM25()) == []
