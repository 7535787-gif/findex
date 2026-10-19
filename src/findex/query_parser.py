
import re
from dataclasses import dataclass

from .index_model import Index


@dataclass(frozen=True)
class Term:
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {
            posting.doc_id
            for posting in index._postings.get(self.value, ())
        }

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Phrase:
    value: str

    def evaluate(self, index: Index) -> set[int]:
        from .tokenize import tokenize

        phrase_terms = list(tokenize(self.value))
        if not phrase_terms:
            return set()

        postings_by_term = [
            {
                posting.doc_id: set(posting.positions)
                for posting in index._postings.get(term, ())
            }
            for term in phrase_terms
        ]

        if any(not postings for postings in postings_by_term):
            return set()

        candidate_docs = set(postings_by_term[0])

        for postings in postings_by_term[1:]:
            candidate_docs &= set(postings)

        matches = set()

        for doc_id in candidate_docs:
            first_positions = postings_by_term[0][doc_id]

            for start in first_positions:
                if all(
                    start + offset in postings_by_term[offset][doc_id]
                    for offset in range(1, len(phrase_terms))
                ):
                    matches.add(doc_id)
                    break

        return matches

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class And:
    left: object
    right: object

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            & self.right.evaluate(index)
        )

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Or:
    left: object
    right: object

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            | self.right.evaluate(index)
        )

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Not:
    node: object

    def evaluate(self, index: Index) -> set[int]:
        return set(index.documents) - self.node.evaluate(index)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


TOKEN_PATTERN = re.compile(
    r'"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()"]+',
    re.IGNORECASE,
)


class Parser:
    def __init__(self, query: str):
        self.tokens = TOKEN_PATTERN.findall(query)
        self.position = 0

    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def take(self):
        token = self.peek()

        if token is None:
            raise ValueError("Unexpected end of query")

        self.position += 1
        return token

    def parse(self):
        if not self.tokens:
            raise ValueError("Empty query")

        node = self.parse_or()

        if self.peek() is not None:
            raise ValueError(f"Unexpected token: {self.peek()}")

        return node

    def parse_or(self):
        node = self.parse_and()

        while self.peek() and self.peek().upper() == "OR":
            self.take()
            node = Or(node, self.parse_and())

        return node

    def parse_and(self):
        node = self.parse_unary()

        while self.peek() is not None:
            token = self.peek()

            if token == ")" or token.upper() == "OR":
                break

            if token.upper() == "AND":
                self.take()

            node = And(node, self.parse_unary())

        return node

    def parse_unary(self):
        token = self.peek()

        if token is None:
            raise ValueError("Missing operand")

        if token.upper() == "NOT":
            self.take()
            return Not(self.parse_unary())

        if token == "(":
            self.take()
            node = self.parse_or()

            if self.peek() != ")":
                raise ValueError("Missing closing parenthesis")

            self.take()
            return node

        if token == ")":
            raise ValueError("Unexpected closing parenthesis")

        token = self.take()

        if token.startswith('"'):
            return Phrase(token[1:-1].lower())

        return Term(token.lower())


def parse_query(query: str):
    return Parser(query).parse()
