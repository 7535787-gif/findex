"""Async frontier/worker crawl API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import httpx

from .core import AsyncCrawler, CrawlPage, CrawlStats, Page


async def crawl(
    seeds: str | Iterable[str],
    *,
    client: httpx.AsyncClient,
    max_pages: int,
    concurrency: int,
    per_host: int,
    allowed_domains: set[str],
    user_agent: str,
    host_delay: float = 0.25,
    timeout: float = 10.0,
    max_depth: int = 3,
) -> AsyncIterator[Page]:
    """Yield pages from one or more seeds using an asyncio worker frontier.

    The caller owns the AsyncClient lifecycle and must close it with an async
    context manager after consuming or closing this generator.
    """
    crawler = AsyncCrawler(
        client,
        user_agent=user_agent,
        concurrency=concurrency,
        per_host=per_host,
        host_delay=host_delay,
        timeout=timeout,
        stats=CrawlStats(),
        allowed_domains=allowed_domains,
    )
    async for page in crawler.crawl(seeds, max_pages=max_pages, max_depth=max_depth):
        yield page


__all__ = ["AsyncCrawler", "CrawlPage", "CrawlStats", "Page", "crawl"]
