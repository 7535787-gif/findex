import argparse
import pickle
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .corpus import iter_documents
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


def build_index(data_dir: Path):
    index = defaultdict(list)
    documents = {}

    for doc_id, document in enumerate(iter_documents(data_dir)):
        counts = Counter(tokenize(document.text))
        length = sum(counts.values())

        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=str(document.path),
            title=document.path.stem,
            length=length,
        )

        for term, tf in counts.items():
            index[term].append(
                Posting(doc_id, tf)
            )

    for postings in index.values():
        postings.sort(
            key=lambda posting: posting.doc_id
        )

    return dict(index), documents


def save_index(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    output_path: Path,
) -> float:
    start = time.perf_counter()

    data = {
        "index": {
            term: [
                (posting.doc_id, posting.tf)
                for posting in postings
            ]
            for term, postings in index.items()
        },
        "documents": {
            doc_id: (
                document.path,
                document.title,
                document.length,
            )
            for doc_id, document in documents.items()
        },
    }

    with output_path.open("wb") as file:
        pickle.dump(data, file)

    return time.perf_counter() - start


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
        # pickle може виконувати довільний код під час десеріалізації.
        data = pickle.load(file)

    index = {
        term: [
            Posting(doc_id, tf)
            for doc_id, tf in postings
        ]
        for term, postings in data["index"].items()
    }

    documents = {
        doc_id: DocMeta(
            doc_id=doc_id,
            path=path,
            title=title,
            length=length,
        )
        for doc_id, (path, title, length)
        in data["documents"].items()
    }

    elapsed = time.perf_counter() - start

    return index, documents, elapsed


def postings_to_ids(
    postings: list[Posting],
) -> list[int]:
    """Перетворює список Posting у список doc_id."""
    return [
        posting.doc_id
        for posting in postings
    ]


# =========================
# MERGE: AND
# =========================

def and_merge(
    left: list[int],
    right: list[int],
) -> list[int]:
    """Перетин двох відсортованих списків через два покажчики."""
    result = []

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
    result = []

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
    result = []

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

    result = postings_to_ids(
        index.get(terms[0], [])
    )

    for term in terms[1:]:
        current = postings_to_ids(
            index.get(term, [])
        )

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
        current = postings_to_ids(
            index.get(term, [])
        )

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

    excluded = postings_to_ids(
        index.get(term, [])
    )

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

    result = set(
        postings_to_ids(
            index.get(terms[0], [])
        )
    )

    for term in terms[1:]:
        current = set(
            postings_to_ids(
                index.get(term, [])
            )
        )

        result &= current

    return sorted(result)


def search_or_set(
    index: dict[str, list[Posting]],
    terms: list[str],
) -> list[int]:
    """OR-пошук через set."""
    result = set()

    for term in terms:
        current = set(
            postings_to_ids(
                index.get(term, [])
            )
        )

        result |= current

    return sorted(result)


def search_not_set(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    term: str,
) -> list[int]:
    """NOT-пошук через set."""
    all_docs = set(documents.keys())

    excluded = set(
        postings_to_ids(
            index.get(term, [])
        )
    )

    return sorted(
        all_docs - excluded
    )


# =========================
# ЗАГАЛЬНИЙ ПОШУК
# =========================

def search(
    index: dict[str, list[Posting]],
    documents: dict[int, DocMeta],
    terms: list[str],
    operation: str = "AND",
    engine: str = "merge",
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
                raise ValueError(
                    "NOT requires exactly one term"
                )

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
                raise ValueError(
                    "NOT requires exactly one term"
                )

            return search_not_set(
                index,
                documents,
                terms[0],
            )

    raise ValueError(
        f"Unknown operation or engine: "
        f"{operation}, {engine}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build and save an inverted index"
    )

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

    index, documents = build_index(
        args.data_dir
    )

    build_time = (
        time.perf_counter() - start
    )

    _, peak = (
    tracemalloc.get_traced_memory()
    )

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

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Terms: {len(index)}"
    )

    print(
        f"Build time: {build_time:.6f} s"
    )

    print(
        f"Build peak memory: "
        f"{peak / 1024 / 1024:.2f} MB"
    )

    print(
        f"Save time: {save_time:.6f} s"
    )

    print(
        f"JSON save time: "
        f"{json_save_time:.6f} s"
    )

    print(
        f"Index saved to: {args.out}"
    )

    print(
        f"JSON index saved to: {json_path}"
    )


if __name__ == "__main__":
    main()