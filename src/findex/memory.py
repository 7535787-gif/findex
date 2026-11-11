"""Compare memory and persistence costs of posting representations."""

import pickle
import tempfile
import time
import tracemalloc
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .index import Posting, build_index

type IndexData = dict[str, list[Posting]]


@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class SlotsPosting:
    doc_id: int
    tf: int


type PlainStorage = list[PlainPosting]
type SlotsStorage = list[SlotsPosting]
type ArrayStorage = dict[str, array[int]]
type Storage = PlainStorage | SlotsStorage | ArrayStorage
type Measurement = tuple[Storage, int]
type MeasureFunction = Callable[[IndexData], Measurement]


def measure_plain(index: IndexData) -> tuple[PlainStorage, int]:
    tracemalloc.start()
    try:
        storage = [
            PlainPosting(posting.doc_id, posting.tf)
            for postings in index.values()
            for posting in postings
        ]
        _, peak = tracemalloc.get_traced_memory()
        return storage, peak
    finally:
        tracemalloc.stop()


def measure_slots(index: IndexData) -> tuple[SlotsStorage, int]:
    tracemalloc.start()
    try:
        storage = [
            SlotsPosting(posting.doc_id, posting.tf)
            for postings in index.values()
            for posting in postings
        ]
        _, peak = tracemalloc.get_traced_memory()
        return storage, peak
    finally:
        tracemalloc.stop()


def measure_array(index: IndexData) -> tuple[ArrayStorage, int]:
    tracemalloc.start()
    try:
        storage: ArrayStorage = {
            term: array(
                "I",
                [
                    value
                    for posting in postings
                    for value in (posting.doc_id, posting.tf)
                ],
            )
            for term, postings in index.items()
        }
        _, peak = tracemalloc.get_traced_memory()
        return storage, peak
    finally:
        tracemalloc.stop()


def measure_file_and_load(storage: Storage, path: Path) -> tuple[int, float, Storage]:
    with path.open("wb") as file:
        pickle.dump(storage, file, protocol=pickle.HIGHEST_PROTOCOL)
    file_size = path.stat().st_size
    start = time.perf_counter()
    with path.open("rb") as file:
        loaded: Storage = pickle.load(file)
    return file_size, time.perf_counter() - start, loaded


def main() -> None:
    index, _documents = build_index(Path("data"))
    variants: list[tuple[str, MeasureFunction]] = [
        ("Plain dataclass", measure_plain),
        ("slots=True", measure_slots),
        ("array('I')", measure_array),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(
            f"{'Representation':<20}"
            f"{'Peak memory (MB)':>20}"
            f"{'File size (KB)':>18}"
            f"{'Load time (s)':>16}"
        )
        print("-" * 74)
        for name, measure in variants:
            storage, peak = measure(index)
            file_path = temp_path / f"{name.replace(' ', '_')}.pkl"
            file_size, load_time, loaded = measure_file_and_load(storage, file_path)
            print(
                f"{name:<20}"
                f"{peak / 1024 / 1024:>20.2f}"
                f"{file_size / 1024:>18.2f}"
                f"{load_time:>16.6f}"
            )
            del storage, loaded


if __name__ == "__main__":
    main()
