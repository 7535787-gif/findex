import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def calculate_stats(root: Path, limit: int | None = None) -> tuple:
    documents = iter_documents(root)

    if limit is not None:
        documents = itertools.islice(documents, limit)

    document_count = 0
    token_count = 0
    vocabulary = Counter()

    for document in documents:
        document_count += 1

        for token in tokenize(document.text):
            token_count += 1
            vocabulary[token] += 1

    return document_count, token_count, vocabulary


def calculate_stats_eager(root: Path, limit: int | None = None) -> tuple:
    documents = list(iter_documents(root))

    if limit is not None:
        documents = documents[:limit]

    token_lists = [
        list(tokenize(document.text))
        for document in documents
    ]

    vocabulary = Counter(
        token
        for tokens in token_lists
        for token in tokens
    )

    token_count = sum(len(tokens) for tokens in token_lists)

    return len(documents), token_count, vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate statistics for a text corpus."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the corpus directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents",
    )

    args = parser.parse_args()

    # Lazy version
    tracemalloc.start()
    start_time = time.perf_counter()

    document_count, token_count, vocabulary = calculate_stats(
        args.root,
        args.limit,
    )

    lazy_elapsed = time.perf_counter() - start_time
    _, lazy_peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=== Lazy version ===")
    print(f"Documents: {document_count}")
    print(f"Tokens: {token_count}")
    print(f"Vocabulary: {len(vocabulary)}")
    print("\nTop 50 tokens:")

    for token, count in vocabulary.most_common(50):
        print(f"{token}: {count}")

    print(f"\nElapsed time: {lazy_elapsed:.4f} seconds")
    print(f"Peak memory: {lazy_peak_memory / 1024 / 1024:.2f} MB")

    # Eager version
    tracemalloc.start()
    start_time = time.perf_counter()

    (
        eager_document_count,
        eager_token_count,
        eager_vocabulary,
    ) = calculate_stats_eager(
        args.root,
        args.limit,
    )

    eager_elapsed = time.perf_counter() - start_time
    _, eager_peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n=== Eager version ===")
    print(f"Documents: {eager_document_count}")
    print(f"Tokens: {eager_token_count}")
    print(f"Vocabulary: {len(eager_vocabulary)}")
    print("\nTop 50 tokens:")

    for token, count in eager_vocabulary.most_common(50):
        print(f"{token}: {count}")

    print(f"\nElapsed time: {eager_elapsed:.4f} seconds")
    print(f"Peak memory: {eager_peak_memory / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()