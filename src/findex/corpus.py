from pathlib import Path
from typing import Iterator, NamedTuple


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Cannot read {path}: {exc}")
            continue

        yield Document(
            doc_id=path.stem,
            path=path,
            text=text,
        )