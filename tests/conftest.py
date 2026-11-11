from pathlib import Path

import pytest

from findex.index import build_index
from findex.index_model import Index


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("Python async search engine", encoding="utf-8")
    (corpus / "two.txt").write_text("Python typing and testing", encoding="utf-8")
    (corpus / "three.txt").write_text("Unicode café Київ", encoding="utf-8")
    return corpus


@pytest.fixture
def tiny_index(tiny_corpus: Path) -> Index:
    postings, documents = build_index(tiny_corpus)
    return Index(postings, documents)
