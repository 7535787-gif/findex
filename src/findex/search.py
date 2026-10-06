import argparse
import time
import tracemalloc
from pathlib import Path

from .index import load_index, search


def main():
    parser = argparse.ArgumentParser(
        description="Search the inverted index"
    )

    parser.add_argument(
        "index_file",
        type=Path,
        help="Path to saved index",
    )

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Search engine",
    )

    parser.add_argument(
    "--operation",
    type=str.upper,
    choices=["AND", "OR", "NOT"],
    default="AND",
    help="Boolean operation",
)

    args = parser.parse_args()
    
    tracemalloc.start()

    index, documents, load_time = load_index(args.index_file)

    terms = args.query.lower().split()

    search_start = time.perf_counter()

    results = search(
        index,
        documents,
        terms,
        operation=args.operation,
        engine=args.engine,
    )

    search_time = time.perf_counter() - search_start

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query: {args.query}")
    print(f"Operation: {args.operation}")
    print(f"Engine: {args.engine}")
    print(f"Results: {results}")
    print(f"Results count: {len(results)}")
    print(f"Load time: {load_time:.6f} s")
    print(f"Search time: {search_time:.6f} s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()