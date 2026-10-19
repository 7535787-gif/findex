
from findex.index import DocMeta, Posting
from findex.index_model import Index


def make_index():
    postings = {
        "python": [Posting(0, 2), Posting(1, 1)],
        "search": [Posting(1, 3)],
    }
    documents = {
        0: DocMeta(0, "doc0.txt", "doc0", 10),
        1: DocMeta(1, "doc1.txt", "doc1", 20),
    }
    return Index(postings, documents)


def test_index_mapping_methods():
    index = make_index()

    assert len(index) == 2
    assert "python" in index
    assert "missing" not in index
    assert index["python"] == (Posting(0, 2), Posting(1, 1))
    assert list(index) == ["python", "search"]
    assert repr(index) == "Index(terms=2, documents=2)"


def test_index_statistics():
    index = make_index()

    assert index.num_docs == 2
    assert index.avg_doc_length == 15
    assert index.df("python") == 2
    assert index.df("search") == 1
    assert index.df("missing") == 0


def test_average_length_is_cached():
    index = make_index()

    first = index.avg_doc_length
    assert "avg_doc_length" in index.__dict__
    assert index.avg_doc_length == first
