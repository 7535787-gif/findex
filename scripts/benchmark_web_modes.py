"""Compare 20 concurrent /search requests for def and async def handlers."""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from findex.index import load_index
from findex.index_model import Index
from findex.web.app import create_app
from findex.web.config import Settings
from findex.web.dependencies import get_index


def measure(index_path: Path, mode: str, requests: int) -> tuple[float, float]:
    postings, documents, _ = load_index(index_path)
    index = Index(postings, documents)
    app = create_app(Settings(index_path=None), search_mode=mode, load_on_startup=False)
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as client:
        # Warm-up avoids counting first-request setup in the sample.
        warmup = client.get("/search", params={"q": "python"})
        if warmup.status_code != 200:
            raise RuntimeError(f"Warm-up failed: {warmup.status_code}")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=requests) as pool:
            responses = list(
                pool.map(
                    lambda _: client.get("/search", params={"q": "python"}),
                    range(requests),
                )
            )
        elapsed = time.perf_counter() - started
    if any(response.status_code != 200 for response in responses):
        raise RuntimeError("One or more benchmark requests failed")
    return elapsed, requests / elapsed if elapsed else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()
    if args.requests != 20:
        parser.error("The laboratory comparison uses exactly 20 concurrent requests")
    for mode in ("async", "def"):
        elapsed, throughput = measure(args.index, mode, args.requests)
        print(
            f"mode={mode:5s} concurrent={args.requests} "
            f"wall_s={elapsed:.6f} requests_per_second={throughput:.2f}"
        )


if __name__ == "__main__":
    main()
