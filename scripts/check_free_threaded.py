r"""Stress-check deterministic threaded indexing on a free-threaded interpreter.

Run from the repository root with:
    $env:PYTHONPATH = "$PWD\src"
    .\.venv313t\Scripts\python.exe scripts\check_free_threaded.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from findex.index import build_index


def fingerprint(index, documents) -> str:
    payload = {
        "index": {
            term: [
                [posting.doc_id, posting.tf, list(posting.positions)]
                for posting in postings
            ]
            for term, postings in sorted(index.items())
        },
        "documents": {
            str(doc_id): [doc.path, doc.title, doc.length]
            for doc_id, doc in sorted(documents.items())
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="findex-free-threaded-") as temp:
        corpus = Path(temp)
        for number in range(32):
            words = [
                "python",
                "concurrency",
                "index",
                f"document{number}",
                "кібербезпека",
                "search",
            ] * (number + 1)
            (corpus / f"doc-{number:02}.txt").write_text(
                " ".join(words),
                encoding="utf-8",
            )

        expected = fingerprint(*build_index(corpus, executor="serial"))
        print(f"Serial baseline: {expected}")

        for workers in (2, 4, 8):
            for repetition in range(1, 6):
                actual = fingerprint(
                    *build_index(corpus, workers=workers, executor="threads")
                )
                if actual != expected:
                    raise SystemExit(
                        f"Mismatch: workers={workers}, run={repetition}"
                    )
                print(
                    f"PASS: threads={workers}, run={repetition}, "
                    f"SHA256={actual}"
                )

    print("PASS: all threaded indexes matched the serial baseline")


if __name__ == "__main__":
    main()
