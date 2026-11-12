"""Compare CPU-bound threads with the interpreter's GIL state."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor


def cpu_work(rounds: int) -> int:
    value = 0
    for number in range(rounds):
        value += (number * number) % 97
    return value


def main() -> None:
    gil_check = getattr(sys, "_is_gil_enabled", None)
    print(f"Python: {sys.version.split()[0]}")
    print(f"sys._is_gil_enabled(): {gil_check() if gil_check else 'unavailable'}")
    rounds = 2_000_000
    for workers in (1, 2, 4, 8):
        start = time.perf_counter()
        if workers == 1:
            cpu_work(rounds)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(cpu_work, [rounds // workers] * workers))
        print(f"threads={workers}: wall={time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
