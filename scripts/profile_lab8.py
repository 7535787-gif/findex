"""Reproducible cProfile runs for full-corpus indexing and three-term search."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from findex.index import build_index, load_index
from findex.index_model import Index
from findex.ranking import BM25, search
from findex.tokenize import tokenize


def profile_call(label, function, output_dir: Path) -> pstats.Stats:
    profile_path = output_dir / f"{label}.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    with (output_dir / f"{label}.txt").open("w", encoding="utf-8") as stream:
        stats.stream = stream
        stats.print_stats(30)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--index", type=Path, default=Path("index.bin"))
    parser.add_argument("--out", type=Path, default=Path("docs/profiling"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        profile_call(
            "index-build-before",
            lambda: build_index(args.corpus, include_positions=True),
            args.out,
        )

    postings, documents, _ = load_index(args.index)
    index = Index(postings, documents)
    query = ["його", "вода", "роки"]
    # Confirm all three tokens are present in the actual indexed corpus.
    query = [term for term in query if term in index]
    if len(query) != 3:
        raise SystemExit(
            "The three-term profiling query must be present in the supplied index."
        )
    profile_call(
        "search-before",
        lambda: [
            search(index, query, BM25(), top_k=10)
            for _ in range(100)
        ],
        args.out,
    )
    print(f"Profiles and pstats summaries written to {args.out}")


if __name__ == "__main__":
    main()
