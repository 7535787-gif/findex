"""Single-request fetch API."""

from __future__ import annotations

import httpx

from .core import AsyncCrawler, FetchResult


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
) -> FetchResult:
    """Fetch one URL with bounded retries and timeout handling.

    The caller must configure a descriptive User-Agent containing a contact URL.
    """
    user_agent = client.headers.get("User-Agent")
    if not user_agent or "(" not in user_agent or ")" not in user_agent:
        raise ValueError(
            "AsyncClient must provide a descriptive User-Agent with contact information"
        )
    crawler = AsyncCrawler(
        client,
        user_agent=user_agent,
        concurrency=1,
        per_host=1,
        host_delay=0,
        timeout=timeout,
    )
    return await crawler.fetch(url)


__all__ = ["FetchResult", "fetch"]
