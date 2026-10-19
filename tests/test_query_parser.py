

from findex.index import DocMeta, Posting
from findex.index_model import Index
from findex.query_parser import (
    And,
    Not,
    Or,
    Phrase,
    Term,
    parse_query,
)


def test_single_term():
    assert parse_query("python") == Term("python")


def test_implicit_and():
    assert parse_query("python async") == And(
        Term("python"), Term("async")
    )


def test_parentheses_and_or():
    assert parse_query("python AND (async OR await)") == And(
        Term("python"),
        Or(Term("async"), Term("await")),
    )


def test_phrase():
    assert parse_query('"event loop"') == Phrase("event loop")


def test_complex_query():
    query = 'python AND (async OR await) NOT java "event loop"'
    assert parse_query(query) == And(
        And(
            And(
                Term("python"),
                Or(Term("async"), Term("await")),
            ),
            Not(Term("java")),
        ),
        Phrase("event loop"),
    )


def test_operators():
    assert (Term("a") & Term("b")) == And(Term("a"), Term("b"))
    assert (Term("a") | Term("b")) == Or(Term("a"), Term("b"))
    assert ~Term("a") == Not(Term("a"))


def test_evaluate_boolean():
    index = Index(
        {
            "python": [Posting(0, 1), Posting(1, 2)],
            "java": [Posting(1, 1)],
        },
        {
            0: DocMeta(0, "a.txt", "a", 10),
            1: DocMeta(1, "b.txt", "b", 20),
        },
    )

    assert parse_query("python NOT java").evaluate(index) == {0}
    assert parse_query("python OR java").evaluate(index) == {0, 1}



def test_phrase_search_uses_positions():
    index = Index(
        {
            "event": [
                Posting(0, 1, (2,)),
                Posting(1, 1, (0,)),
            ],
            "loop": [
                Posting(0, 1, (3,)),
                Posting(1, 1, (2,)),
            ],
        },
        {
            0: DocMeta(0, "a.txt", "a", 10),
            1: DocMeta(1, "b.txt", "b", 10),
        },
    )

    assert parse_query('"event loop"').evaluate(index) == {0}

