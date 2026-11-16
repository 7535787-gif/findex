import argparse
import logging
import multiprocessing
import pickle
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from .corpus import iter_documents
from .timing import timed
from .tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int
    content_path: str | None = None
    line_offset: int | None = None

    def __getstate__(self) -> tuple[object, ...]:
        return (
            self.doc_id,
            self.path,
            self.title,
            self.length,
            self.content_path,
            self.line_offset,
        )

    def __setstate__(self, state: tuple[object, ...] | list[object]) -> None:
        values = list(state)
        values.extend([None] * (6 - len(values)))
        object.__setattr__(self, "doc_id", values[0])
        object.__setattr__(self, "path", values[1])
        object.__setattr__(self, "title", values[2])
        object.__setattr__(self, "length", values[3])
        object.__setattr__(self, "content_path", values[4])
        object.__setattr__(self, "line_offset", values[5])


class SerializedIndexV2(TypedDict):
    doc_ids: NDArray[np.int32]
    tfs: NDArray[np.int32]
    offsets: dict[str, tuple[int, int]]
    positions: dict[str, list[tuple[int, ...]]]


class SerializedIndex(TypedDict):
    format_version: int
    index: SerializedIndexV2
    doc_lengths: NDArray[np.int32]
    doc_length_ids: NDArray[np.int32]
    documents: dict[int, tuple[str, str, int, str | None, int | None]]


def build_partial(
    file_paths: tuple[str, ...],
    start_doc_id: int,
    include_positions: bool = True,
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Build one independent index shard; safe to call in a spawned process."""
    index: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}

    for offset, raw_path in enumerate(file_paths):
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise OSError(f"Worker cannot read {path}") from exc
        tokens = list(tokenize(text))
        counts = Counter(tokens)
        doc_id = start_doc_id + offset
        documents[doc_id] = DocMeta(
            doc_id=doc_id, path=str(path), title=path.stem, length=len(tokens)
        )
        positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
        if include_positions:
            for position, term in enumerate(tokens):
                positions_by_term[term].append(position)
        for term, frequency in counts.items():
            index[term].append(
                Posting(doc_id, frequency, tuple(positions_by_term[term]))
            )
    return dict(index), documents


def merge_partials(
    partials: Iterable[tuple[dict[str, list[Posting]], dict[int, DocMeta]]],
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Merge independent shards into one deterministically ordered index."""
    merged: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}
    for partial_index, partial_documents in partials:
        merged_docs = documents.keys() & partial_documents.keys()
        if merged_docs:
            raise ValueError(
                f"Duplicate document IDs in partial indexes: {merged_docs}"
            )
        documents.update(partial_documents)
        for term, postings in partial_index.items():
            merged[term].extend(postings)
    for postings in merged.values():
        postings.sort(key=lambda posting: posting.doc_id)
    return dict(merged), dict(sorted(documents.items()))


def merge(
    partials: Iterable[tuple[dict[str, list[Posting]], dict[int, DocMeta]]],
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Public name for merging partial indexes."""
    return merge_partials(partials)


def build_index(
    data_dir: Path,
    progress_callback: Callable[[int], None] | None = None,
    *,
    limit: int | None = None,
    include_positions: bool = True,
    workers: int = 1,
    executor: Literal["serial", "threads", "processes"] = "serial",
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Build an index serially or with thread/process workers."""
    if limit is not None and limit < 1:
        raise ValueError("Document limit must be greater than zero")
    if data_dir.is_file() and data_dir.suffix.lower() == ".jsonl":
        if executor != "serial" or workers != 1:
            raise ValueError(
                "JSONL crawling output is indexed in streaming serial mode"
            )
        return build_index_from_jsonl(
            data_dir,
            progress_callback=progress_callback,
            limit=limit,
            include_positions=include_positions,
        )
    if workers < 1:
        raise ValueError("Workers must be greater than zero")
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unknown executor: {executor}")
    paths = list(data_dir.rglob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        return {}, {}

    # Each shard receives a contiguous, globally unique doc_id range.
    shard_count = min(workers if executor != "serial" else 1, len(paths))
    chunk_size = (len(paths) + shard_count - 1) // shard_count
    chunks = [
        tuple(str(path) for path in paths[i : i + chunk_size])
        for i in range(0, len(paths), chunk_size)
    ]
    tasks = [
        (chunk, sum(len(item) for item in chunks[:i]), include_positions)
        for i, chunk in enumerate(chunks)
    ]

    if executor == "serial":
        partials = [build_partial(*task) for task in tasks]
    else:
        if executor == "threads":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                partials = list(pool.map(_build_partial_task, tasks))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                partials = list(pool.map(_build_partial_task, tasks))
    if progress_callback is not None:
        progress_callback(len(paths))
    return merge_partials(partials)


def build_index_from_jsonl(
    jsonl_path: Path,
    progress_callback: Callable[[int], None] | None = None,
    *,
    limit: int | None = None,
    include_positions: bool = True,
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Build an index lazily from crawler JSONL without loading all pages first."""
    index: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}
    for doc_id, document in enumerate(iter_documents(jsonl_path)):
        if limit is not None and doc_id >= limit:
            break
        tokens = list(tokenize(document.text))
        counts = Counter(tokens)
        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=document.doc_id,
            title=document.title or document.doc_id,
            length=len(tokens),
            content_path=str(jsonl_path),
            line_offset=document.offset,
        )
        positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
        if include_positions:
            for position, term in enumerate(tokens):
                positions_by_term[term].append(position)
        for term, frequency in counts.items():
            index[term].append(
                Posting(doc_id, frequency, tuple(positions_by_term[term]))
            )
        if progress_callback is not None:
            progress_callback(doc_id + 1)
    return dict(index), documents


def _build_partial_task(
    task: tuple[tuple[str, ...], int, bool],
) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Top-level adapter keeps the submitted callable spawn-picklable."""
    return build_partial(*task)


@timed
def save_index(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    output_path: Path,
) -> float:
    start = time.perf_counter()

    data: SerializedIndex = {
        "format_version": 2,
        "index": {
            "doc_ids": np.fromiter(
                (posting.doc_id for postings in index.values() for posting in postings),
                dtype=np.int32,
            ),
            "tfs": np.fromiter(
                (posting.tf for postings in index.values() for posting in postings),
                dtype=np.int32,
            ),
            "offsets": {},
            "positions": {},
        },
        "doc_lengths": np.asarray(
            [documents[doc_id].length for doc_id in sorted(documents)],
            dtype=np.int32,
        ),
        "doc_length_ids": np.asarray(sorted(documents), dtype=np.int32),
        "documents": {
            doc_id: (
                document.path,
                document.title,
                document.length,
                document.content_path,
                document.line_offset,
            )
            for doc_id, document in documents.items()
        },
    }

    cursor = 0
    for term, postings in index.items():
        stop = cursor + len(postings)
        data["index"]["offsets"][term] = (cursor, stop)
        data["index"]["positions"][term] = [posting.positions for posting in postings]
        cursor = stop

    with output_path.open("wb") as file:
        pickle.dump(data, file)

    return time.perf_counter() - start


@timed
def load_index(
    input_path: Path,
) -> tuple[
    dict[str, list[Posting]],
    dict[int, DocMeta],
    float,
]:
    start = time.perf_counter()

    with input_path.open("rb") as file:
        # Не завантажуйте pickle-файли з ненадійних джерел:
        # pickle може виконувати довільний код.
        data: dict[str, Any] = pickle.load(file)

    index: dict[str, list[Posting]] = {}
    index_payload = data["index"]
    if data.get("format_version") == 2:
        all_doc_ids = np.asarray(index_payload["doc_ids"], dtype=np.int32)
        all_frequencies = np.asarray(index_payload["tfs"], dtype=np.int32)
        for term, (start, stop) in index_payload["offsets"].items():
            positions_by_posting = index_payload["positions"].get(
                term, [()] * (stop - start)
            )
            index[term] = [
                Posting(int(doc_id), int(tf), tuple(positions))
                for doc_id, tf, positions in zip(
                    all_doc_ids[start:stop],
                    all_frequencies[start:stop],
                    positions_by_posting,
                    strict=True,
                )
            ]
    else:
        for term, postings in index_payload.items():
            restored_postings: list[Posting] = []
            if isinstance(postings, dict):
                # Read an intermediate per-term array format as well.
                posting_data = cast(dict[str, Any], postings)

                doc_ids: NDArray[np.int32] = np.asarray(
                    posting_data["doc_ids"], dtype=np.int32
                )
                frequencies: NDArray[np.int32] = np.asarray(
                    posting_data["tfs"], dtype=np.int32
                )

                raw_positions: Any = posting_data.get("positions")
                positions_by_posting: list[tuple[int, ...]] = (
                    [tuple(map(int, positions)) for positions in raw_positions]
                    if raw_positions is not None
                    else [()] * len(doc_ids)
                )

                restored_postings = [
                    Posting(int(doc_id), int(tf), positions)
                    for doc_id, tf, positions in zip(
                        doc_ids, frequencies, positions_by_posting, strict=True
                    )
                ]
            else:
                # Backward compatibility with indexes created in Labs 1–7.
                for item in postings:
                    doc_id, tf, *extra = item
                    positions = tuple(extra[0]) if extra else ()
                    restored_postings.append(
                        Posting(doc_id=doc_id, tf=tf, positions=positions)
                    )
            index[term] = restored_postings

    documents = {
        doc_id: DocMeta(
            doc_id=doc_id,
            path=path,
            title=title,
            length=length,
            content_path=extra[0] if len(extra) > 0 else None,
            line_offset=extra[1] if len(extra) > 1 else None,
        )
        for doc_id, (path, title, length, *extra) in data["documents"].items()
    }

    elapsed = time.perf_counter() - start
    return index, documents, elapsed


def postings_to_ids(
    postings: list[Posting],
) -> list[int]:
    """Перетворює список Posting у список doc_id."""
    return [posting.doc_id for posting in postings]


# =========================
# MERGE: AND
# =========================


def and_merge(
    left: list[int],
    right: list[int],
) -> list[int]:
    """Перетин двох відсортованих списків через два покажчики."""
    result: list[int] = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1

    return result


# =========================
# MERGE: OR
# =========================


def or_merge(
    left: list[int],
    right: list[int],
) -> list[int]:
    """Об'єднання двох відсортованих списків через два покажчики."""
    result: list[int] = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


# =========================
# MERGE: NOT
# =========================


def not_merge(
    all_docs: list[int],
    excluded: list[int],
) -> list[int]:
    """Виключає excluded з all_docs через два покажчики."""
    result: list[int] = []

    i = 0
    j = 0

    while i < len(all_docs) and j < len(excluded):
        if all_docs[i] == excluded[j]:
            i += 1
            j += 1
        elif all_docs[i] < excluded[j]:
            result.append(all_docs[i])
            i += 1
        else:
            j += 1

    result.extend(all_docs[i:])

    return result


# =========================
# MERGE SEARCH
# =========================


def search_and(
    index: dict[str, list[Posting]],
    terms: list[str],
) -> list[int]:
    """AND-пошук через merge."""
    if not terms:
        return []

    result = postings_to_ids(index.get(terms[0], []))

    for term in terms[1:]:
        current = postings_to_ids(index.get(term, []))

        result = and_merge(
            result,
            current,
        )

    return result


def search_or(
    index: dict[str, list[Posting]],
    terms: list[str],
) -> list[int]:
    """OR-пошук через merge."""
    result = []

    for term in terms:
        current = postings_to_ids(index.get(term, []))

        result = or_merge(
            result,
            current,
        )

    return result


def search_not(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    term: str,
) -> list[int]:
    """NOT-пошук через merge."""
    all_docs = sorted(documents.keys())

    excluded = postings_to_ids(index.get(term, []))

    return not_merge(
        all_docs,
        excluded,
    )


# =========================
# SET SEARCH
# =========================


def search_and_set(
    index: dict[str, list[Posting]],
    terms: list[str],
) -> list[int]:
    """AND-пошук через set."""
    if not terms:
        return []

    result = set(postings_to_ids(index.get(terms[0], [])))

    for term in terms[1:]:
        current = set(postings_to_ids(index.get(term, [])))

        result &= current

    return sorted(result)


def search_or_set(
    index: dict[str, list[Posting]],
    terms: list[str],
) -> list[int]:
    """OR-пошук через set."""
    result: set[int] = set()

    for term in terms:
        current = set(postings_to_ids(index.get(term, [])))

        result |= current

    return sorted(result)


def search_not_set(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    term: str,
) -> list[int]:
    """NOT-пошук через set."""
    all_docs = set(documents.keys())

    excluded = set(postings_to_ids(index.get(term, [])))

    return sorted(all_docs - excluded)


# =========================
# ЗАГАЛЬНИЙ ПОШУК
# =========================


def search(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    terms: list[str],
    operation: Literal["AND", "OR", "NOT"] = "AND",
    engine: Literal["merge", "set"] = "merge",
) -> list[int]:

    if engine == "merge":
        if operation == "AND":
            return search_and(
                index,
                terms,
            )

        elif operation == "OR":
            return search_or(
                index,
                terms,
            )

        elif operation == "NOT":
            if len(terms) != 1:
                raise ValueError("NOT requires exactly one term")

            return search_not(
                index,
                documents,
                terms[0],
            )

    elif engine == "set":
        if operation == "AND":
            return search_and_set(
                index,
                terms,
            )

        elif operation == "OR":
            return search_or_set(
                index,
                terms,
            )

        elif operation == "NOT":
            if len(terms) != 1:
                raise ValueError("NOT requires exactly one term")

            return search_not_set(
                index,
                documents,
                terms[0],
            )

    raise ValueError(f"Unknown operation or engine: {operation}, {engine}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save an inverted index")

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the corpus directory",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output index file",
    )

    args = parser.parse_args()

    tracemalloc.start()

    start = time.perf_counter()

    index, documents = build_index(args.data_dir)

    build_time = time.perf_counter() - start

    _, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # Зберігаємо pickle
    save_time = save_index(
        index,
        documents,
        args.out,
    )

    # Зберігаємо JSON
    from .store import save_json

    json_path = args.out.with_suffix(".json")

    json_save_time = save_json(
        index,
        documents,
        json_path,
    )

    logger.info("Documents: %s", len(documents))

    logger.info("Terms: %s", len(index))

    logger.info("Build time: %.6f s", build_time)

    logger.info("Build peak memory: %.2f MB", peak / 1024 / 1024)

    logger.info("Save time: %.6f s", save_time)

    logger.info("JSON save time: %.6f s", json_save_time)

    logger.info("Index saved to: %s", args.out)

    logger.info("JSON index saved to: %s", json_path)


if __name__ == "__main__":
    main()
