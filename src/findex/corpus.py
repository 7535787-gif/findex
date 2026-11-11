"""Lazy iteration over text documents."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            continue
        yield Document(doc_id=path.stem, path=path, text=text)
