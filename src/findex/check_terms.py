
from pathlib import Path

from .index_model import open_index
from .tokenize import tokenize

with open_index(Path("index.bin")) as index:
    print("Sample terms:", list(index)[:40])

    query_terms = list(tokenize("python programming"))
    print("Query tokens:", query_terms)
    print("Matching terms:", [term for term in query_terms if term in index])
