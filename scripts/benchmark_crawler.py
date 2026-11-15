"""Benchmark crawler concurrency on one authorized site."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import threading
import time
from pathlib import Path

import httpx
import psutil

from findex.crawler import AsyncCrawler, CrawlStats, write_jsonl_page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark asyncio crawler concurrency"
    )
    parser.add_argument("start_url")
    parser.add_argument("--contact", required=True)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--per-host", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("crawler_benchmark.csv"))
    args = parser.parse_args()
    if args.max_pages < 200:
        parser.error("--max-pages must be at least 200 for the lab benchmark")
    if "@" not in args.contact:
        parser.error("--contact must be a valid contact email")

    crawl_logger = logging.getLogger("findex.crawl")
    crawl_logger.setLevel(logging.INFO)
    handler = logging.FileHandler("crawl.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    crawl_logger.addHandler(handler)

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.wait(0.05):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    rows: list[dict[str, object]] = []
    for concurrency in (1, 5, 20):
        stats = CrawlStats()
        peak_rss[0] = process.memory_info().rss
        stop.clear()
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        output_path = Path(f"crawl-{concurrency}.jsonl")
        started = time.perf_counter()

        async def run(
            concurrency: int = concurrency,
            stats: CrawlStats = stats,
            output_path: Path = output_path,
        ) -> None:
            user_agent = f"findex-lab6-benchmark/1.0 (+mailto:{args.contact})"
            async with httpx.AsyncClient(
                follow_redirects=False,
                max_redirects=10,
                headers={"Accept": "text/html,application/xhtml+xml"},
            ) as client:
                crawler = AsyncCrawler(
                    client,
                    user_agent=user_agent,
                    concurrency=concurrency,
                    per_host=args.per_host,
                    host_delay=args.delay,
                    timeout=args.timeout,
                    stats=stats,
                )
                stream = await asyncio.to_thread(
                    output_path.open, "w", encoding="utf-8", newline="\n"
                )
                try:
                    async for page in crawler.crawl(
                        args.start_url,
                        max_pages=args.max_pages,
                        max_depth=args.max_depth,
                    ):
                        await asyncio.to_thread(write_jsonl_page, stream, page)
                        await asyncio.to_thread(stream.flush)
                finally:
                    await asyncio.to_thread(stream.close)

        try:
            asyncio.run(run())
        finally:
            elapsed = time.perf_counter() - started
            stop.set()
            sampler.join()
        rows.append(
            {
                "concurrency": concurrency,
                "wall_s": round(elapsed, 3),
                "pages": stats.pages,
                "pages_per_second": round(stats.pages / elapsed, 3) if elapsed else 0,
                "errors": stats.errors,
                "peak_rss_mb": round(peak_rss[0] / (1024 * 1024), 2),
                "per_host": args.per_host,
                "host_delay_s": args.delay,
                "output": str(output_path),
            }
        )
        print(rows[-1])

    crawl_logger.removeHandler(handler)
    handler.close()
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
