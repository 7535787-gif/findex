import tracemalloc
from array import array
from dataclasses import dataclass
from pathlib import Path

from .index import build_index


@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class SlotsPosting:
    doc_id: int
    tf: int


def measure_plain(index):
    tracemalloc.start()

    _storage = [
        PlainPosting(
            posting.doc_id,
            posting.tf,
        )
        for postings in index.values()
        for posting in postings
    ]

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak


def measure_slots(index):
    tracemalloc.start()

    _storage = [
        SlotsPosting(
            posting.doc_id,
            posting.tf,
        )
        for postings in index.values()
        for posting in postings
    ]

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak


def measure_array(index):
    tracemalloc.start()

    _storage = {
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
    tracemalloc.stop()

    return peak


def main():
    index, _ = build_index(Path("data"))

    plain_memory = measure_plain(index)
    slots_memory = measure_slots(index)
    array_memory = measure_array(index)

    print(
        f"Plain dataclass: "
        f"{plain_memory / 1024 / 1024:.2f} MB"
    )

    print(
        f"slots=True: "
        f"{slots_memory / 1024 / 1024:.2f} MB"
    )

    print(
        f"array('I'): "
        f"{array_memory / 1024 / 1024:.2f} MB"
    )


if __name__ == "__main__":
    main()