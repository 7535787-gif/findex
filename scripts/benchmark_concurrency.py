"""Repeatable serial/thread/process indexing benchmark.

Run:
    uv run python scripts/benchmark_concurrency.py data --repeats 3

Free-threaded Python:
    python scripts/benchmark_concurrency.py data --repeats 3 \
        --executors serial,threads \
        --output benchmark_results_313t.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import statistics
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil

from findex.index import build_partial, merge_partials


def _task(
    task: tuple[tuple[str, ...], int, bool],
) -> tuple[Any, Any]:
    """Top-level worker function, safe for multiprocessing spawn."""
    return build_partial(*task)


def _sample_process(
    process: psutil.Process,
    child_cpu: dict[int, float],
    state: dict[str, float],
) -> None:
    """Sample parent RSS and cumulative CPU time of child processes."""
    try:
        state["rss_peak"] = max(
            state["rss_peak"],
            float(process.memory_info().rss),
        )
    except psutil.NoSuchProcess, psutil.AccessDenied:
        pass

    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess, psutil.AccessDenied:
        children = []

    for child in children:
        try:
            cpu_times = child.cpu_times()
            total_cpu = cpu_times.user + cpu_times.system
            child_cpu[child.pid] = max(
                child_cpu.get(child.pid, 0.0),
                total_cpu,
            )
        except psutil.NoSuchProcess, psutil.AccessDenied:
            continue


def run_once(
    paths: list[Path],
    workers: int,
    executor: str,
) -> dict[str, float]:
    """Run one indexing benchmark and collect performance metrics."""
    if workers < 1:
        raise ValueError("Workers must be greater than zero")

    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unknown executor: {executor}")

    worker_count = min(
        workers if executor != "serial" else 1,
        len(paths),
    )

    chunk_size = (len(paths) + worker_count - 1) // worker_count

    chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]

    tasks = [
        (
            tuple(str(path) for path in chunk),
            sum(len(previous) for previous in chunks[:i]),
            True,
        )
        for i, chunk in enumerate(chunks)
    ]

    process = psutil.Process()
    child_cpu: dict[int, float] = {}

    state = {
        "rss_peak": float(process.memory_info().rss),
    }

    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.is_set():
            _sample_process(process, child_cpu, state)
            stop_monitor.wait(0.01)

        _sample_process(process, child_cpu, state)

    parent_cpu_start = time.process_time()
    wall_start = time.perf_counter()

    monitor_thread = threading.Thread(
        target=monitor,
        name="benchmark-monitor",
        daemon=True,
    )
    monitor_thread.start()

    try:
        if executor == "serial":
            partials = [build_partial(*task) for task in tasks]

        elif executor == "threads":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                partials = list(pool.map(_task, tasks))

        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                partials = list(pool.map(_task, tasks))

        merge_start = time.perf_counter()
        merge_partials(partials)
        merge_seconds = time.perf_counter() - merge_start

    finally:
        stop_monitor.set()
        monitor_thread.join()

    wall_seconds = time.perf_counter() - wall_start

    parent_cpu = time.process_time() - parent_cpu_start
    children_cpu = sum(child_cpu.values())
    total_cpu = parent_cpu + children_cpu

    return {
        "wall": wall_seconds,
        "cpu": total_cpu,
        "rss_mb": state["rss_peak"] / (1024 * 1024),
        "merge": merge_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark serial, thread and process indexing."
    )

    parser.add_argument(
        "corpus",
        type=Path,
        help="Directory containing .txt documents",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="CSV output file",
    )

    parser.add_argument(
        "--executors",
        type=str,
        default="serial,threads,processes",
        help="Comma-separated executors: serial,threads,processes",
    )

    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    if not args.corpus.is_dir():
        parser.error(f"Corpus directory does not exist: {args.corpus}")

    executors = [item.strip() for item in args.executors.split(",") if item.strip()]

    allowed = {"serial", "threads", "processes"}

    if not executors or any(item not in allowed for item in executors):
        parser.error("--executors must contain serial, threads and/or processes")

    if len(set(executors)) != len(executors):
        parser.error("--executors must not contain duplicates")

    paths = sorted(args.corpus.rglob("*.txt"))

    if not paths:
        parser.error("Corpus contains no .txt files")

    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})

    print(f"Documents: {len(paths)}")
    print(f"CPU count: {cpu_count}")
    print(f"Repeats: {args.repeats}")
    print(f"Executors: {', '.join(executors)}")
    print()

    configurations: list[tuple[str, int]] = []

    for backend in executors:
        if backend == "serial":
            configurations.append(("serial", 1))
        else:
            configurations.extend((backend, workers) for workers in worker_counts)

    rows: list[dict[str, object]] = []
    baseline: float | None = None

    for backend, workers in configurations:
        print(f"Running {backend}, workers={workers}...")

        samples = [run_once(paths, workers, backend) for _ in range(args.repeats)]

        medians = {
            key: statistics.median(sample[key] for sample in samples)
            for key in samples[0]
        }

        if backend == "serial":
            baseline = medians["wall"]

        speedup = (
            baseline / medians["wall"]
            if baseline is not None and medians["wall"] > 0
            else 1.0
        )

        row: dict[str, object] = {
            "executor": backend,
            "workers": workers,
            "wall_s": round(medians["wall"], 4),
            "cpu_s": round(medians["cpu"], 4),
            "parent_peak_rss_mb": round(medians["rss_mb"], 2),
            "merge_s": round(medians["merge"], 4),
            "speedup": round(speedup, 3),
        }

        rows.append(row)

        print(
            f"  wall={medians['wall']:.2f}s, "
            f"cpu={medians['cpu']:.2f}s, "
            f"RSS={medians['rss_mb']:.2f}MB, "
            f"merge={medians['merge']:.2f}s, "
            f"speedup={speedup:.3f}x"
        )

    fieldnames = [
        "executor",
        "workers",
        "wall_s",
        "cpu_s",
        "parent_peak_rss_mb",
        "merge_s",
        "speedup",
    ]

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Benchmark results saved to: {args.output}")
    print()
    print(",".join(fieldnames))

    for row in rows:
        print(",".join(str(row[field]) for field in fieldnames))


if __name__ == "__main__":
    main()
