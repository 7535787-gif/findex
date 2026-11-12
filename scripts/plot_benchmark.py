"""Plot indexing benchmark speedup."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot indexing benchmark speedup.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_speedup.png"),
    )
    args = parser.parse_args()

    with args.input.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    serial_rows = [row for row in rows if row["executor"] == "serial"]
    if not serial_rows:
        parser.error("CSV does not contain serial baseline")

    baseline = float(serial_rows[0]["wall_s"])

    fig, ax = plt.subplots(figsize=(9, 6))

    for backend in ("threads", "processes"):
        backend_rows = sorted(
            (row for row in rows if row["executor"] == backend),
            key=lambda row: int(row["workers"]),
        )

        if not backend_rows:
            continue

        workers = [int(row["workers"]) for row in backend_rows]
        speedups = [baseline / float(row["wall_s"]) for row in backend_rows]

        ax.plot(
            workers,
            speedups,
            marker="o",
            label=backend,
        )

    max_workers = max(int(row["workers"]) for row in rows)
    ideal_workers = list(range(1, max_workers + 1))

    ax.plot(
        ideal_workers,
        ideal_workers,
        linestyle="--",
        label="Ideal linear speedup",
    )

    ax.axhline(
        1.0,
        linestyle=":",
        label="Serial baseline",
    )

    ax.set_title("Indexing speedup vs. worker count")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup")
    ax.set_xticks(sorted({1, 2, 4, 8, max_workers}))
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Graph saved to: {args.output}")


if __name__ == "__main__":
    main()
