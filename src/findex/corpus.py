"""Lazy iteration over text files and crawler JSONL documents."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str
    title: str | None = None
    offset: int | None = None


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents lazily from a .txt directory or crawler .jsonl file."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".jsonl":
        with root.open("rb") as source:
            while True:
                offset = source.tell()
                raw_line = source.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    url = str(record["url"])
                    text = str(record.get("text", ""))
                    title = str(record.get("title") or url)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                ) as exc:
                    logger.warning(
                        "Skipping invalid JSONL record at byte %d: %s", offset, exc
                    )
                    continue
                yield Document(
                    doc_id=url,
                    path=root,
                    text=text,
                    title=title,
                    offset=offset,
                )
        return

    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            continue
        yield Document(doc_id=path.stem, path=path, text=text)
