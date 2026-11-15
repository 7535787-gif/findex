import asyncio
import json
import logging
from pathlib import Path

import httpx

from findex.corpus import iter_documents
from findex.crawler import (
    AsyncCrawler,
    CrawlPage,
    CrawlStats,
    canonicalize_url,
    write_jsonl_page,
)
from findex.index import build_index, load_index, save_index
from findex.index_model import Index
from findex.ranking import BM25, search


def test_canonicalize_url_normalizes_host_fragment_slash_and_query() -> None:
    assert canonicalize_url("HTTPS://Example.COM/a/?z=2&a=1#section") == (
        "https://example.com/a?a=1&z=2"
    )
    assert canonicalize_url("https://example.com") == "https://example.com/"


def test_fetch_retries_server_errors_but_not_other_4xx() -> None:
    calls = {"/retry": 0, "/missing": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        calls[path] += 1
        if path == "/retry" and calls[path] == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        if path == "/missing":
            return httpx.Response(404)
        return httpx.Response(200, text="ok")

    async def no_sleep(_: float) -> None:
        return None

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-test",
                concurrency=2,
                per_host=2,
                host_delay=0,
                sleep=no_sleep,
                random_value=lambda: 0,
            )
            retry = await crawler.fetch("https://example.test/retry")
            missing = await crawler.fetch("https://example.test/missing")
        assert retry.status == 200
        assert retry.attempts == 2
        assert missing.status == 404
        assert missing.attempts == 1

    asyncio.run(run())
    assert calls == {"/retry": 2, "/missing": 1}


def test_crawl_respects_robots_and_page_cap_and_yields_pages() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if path == "/robots.txt":
            return httpx.Response(
                200, text="User-agent: *\nDisallow: /blocked\nAllow: /"
            )
        if path == "/":
            body = (
                "<html><title>Root</title><body>Hello "
                '<a href="/a#x">A</a><a href="/b/">B</a>'
                '<a href="/blocked">No</a><a href="/c">C</a>'
                "</body></html>"
            )
        else:
            body = f"<html><title>{path}</title><body>text {path}</body></html>"
        return httpx.Response(
            200, text=body, headers={"content-type": "text/html; charset=utf-8"}
        )

    async def run() -> tuple[list[object], CrawlStats]:
        stats = CrawlStats()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-test",
                concurrency=3,
                per_host=3,
                host_delay=0,
                stats=stats,
            )
            pages = [
                page
                async for page in crawler.crawl("https://example.test/", max_pages=4)
            ]
        return pages, stats

    pages, stats = asyncio.run(run())
    assert len(pages) == 3
    assert stats.pages == 3
    assert {page.url for page in pages} == {
        "https://example.test/",
        "https://example.test/a",
        "https://example.test/b",
    }
    assert "/blocked" not in requests
    assert requests.count("/robots.txt") == 1


def test_jsonl_is_read_lazily_and_can_be_indexed(tmp_path: Path) -> None:
    output = tmp_path / "crawl.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        write_jsonl_page(
            stream,
            CrawlPage(
                "https://example.test/one",
                "One",
                "Python asyncio crawler",
                200,
                24,
                0.01,
                0,
            ),
        )
        write_jsonl_page(
            stream,
            CrawlPage(
                "https://example.test/two",
                "Two",
                "Python concurrency",
                200,
                18,
                0.02,
                1,
            ),
        )

    documents = list(iter_documents(output))
    assert len(documents) == 2
    assert documents[0].title == "One"
    assert documents[0].offset == 0
    first_record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["fetched_at"]

    postings, metadata = build_index(output)
    index = Index(postings, metadata)
    results = search(index, ["asyncio"], BM25())
    assert len(results) == 1
    assert results[0].path == "https://example.test/one"
    assert "**asyncio**" in results[0].snippet

    saved = tmp_path / "index.bin"
    save_index(postings, metadata, saved)
    assert saved.is_file()
    loaded_postings, loaded_documents, _ = load_index(saved)
    loaded_results = search(
        Index(loaded_postings, loaded_documents), ["asyncio"], BM25()
    )
    assert loaded_results[0].path == "https://example.test/one"
    assert "**asyncio**" in loaded_results[0].snippet


def test_fetch_retries_timeouts_at_most_three_times() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls += 1
        if calls <= 3:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, text="recovered")

    async def no_sleep(_: float) -> None:
        return None

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-test",
                concurrency=1,
                per_host=1,
                host_delay=0,
                sleep=no_sleep,
                random_value=lambda: 0,
            )
            result = await crawler.fetch("https://example.test/timeout")
        assert result.status == 200
        assert result.attempts == 4
        assert result.error is None

    asyncio.run(run())
    assert calls == 4


def test_redirect_loop_is_reported_without_retry(caplog) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls += 1
        return httpx.Response(302, headers={"location": str(request.url)})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
        ) as client:
            crawler = AsyncCrawler(client, user_agent="findex-test", host_delay=0)
            result = await crawler.fetch("https://example.test/loop")
        assert result.error is not None
        assert "redirect loop" in result.error
        assert result.attempts == 1

    with caplog.at_level(logging.INFO, logger="findex.crawl"):
        asyncio.run(run())
    assert calls == 1
    assert "status=302" in caplog.text
    assert "error=redirect loop" in caplog.text


def test_fetch_retries_429_and_honors_retry_after() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, text="ok")

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-test (+https://example.test/contact)",
                host_delay=0,
                sleep=record_sleep,
            )
            result = await crawler.fetch("https://example.test/rate-limited")
        assert result.status == 200
        assert result.attempts == 2
        assert result.error is None

    asyncio.run(run())
    assert calls == 2
    assert delays == [2.0]


def test_fetch_does_not_retry_non_429_client_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls += 1
        return httpx.Response(403)

    async def no_sleep(_: float) -> None:
        raise AssertionError("A 4xx response must not be retried")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client, user_agent="findex-test", host_delay=0, sleep=no_sleep
            )
            result = await crawler.fetch("https://example.test/forbidden")
        assert result.status == 403
        assert result.attempts == 1
        assert result.error == "HTTP 403"

    asyncio.run(run())
    assert calls == 1


def test_request_gate_per_host_limit_is_shared_across_ports() -> None:
    from findex.crawler import RequestGate

    class FakeClient:
        def __init__(self) -> None:
            self.active = 0
            self.peak = 0

        async def get(self, url: str, timeout: float) -> httpx.Response:
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return httpx.Response(200, request=httpx.Request("GET", url))

    async def run() -> int:
        client = FakeClient()
        gate = RequestGate(concurrency=5, per_host=1, host_delay=0)
        await asyncio.gather(
            gate.get(client, "https://example.test:443/a", 1),
            gate.get(client, "https://example.test:8443/b", 1),
        )
        return client.peak

    assert asyncio.run(run()) == 1


def test_crawler_sends_honest_user_agent_and_respects_allowed_domains() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("user-agent", "")))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/":
            return httpx.Response(
                200,
                text='<a href="https://outside.test/secret">outside</a>',
                headers={"content-type": "text/html"},
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    async def run() -> list[CrawlPage]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-lab6/1.0 (+mailto:student@example.org)",
                host_delay=0,
                allowed_domains={"example.test"},
            )
            return [
                page
                async for page in crawler.crawl("https://example.test/", max_pages=10)
            ]

    pages = asyncio.run(run())
    assert len(pages) == 1
    assert all("mailto:student@example.org" in ua for _, ua in seen)
    assert not any(host == "outside.test" for host, _ in seen)


def test_crawl_accepts_multiple_seeds_and_emits_fetched_at() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            text=f"<html><title>{request.url.path}</title><body>page</body></html>",
            headers={"content-type": "text/html"},
        )

    async def run() -> list[CrawlPage]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(client, user_agent="findex-test", host_delay=0)
            return [
                page
                async for page in crawler.crawl(
                    ["https://example.test/a", "https://example.test/b"],
                    max_pages=2,
                )
            ]

    pages = asyncio.run(run())
    assert {page.url for page in pages} == {
        "https://example.test/a",
        "https://example.test/b",
    }
    assert all(page.fetched_at for page in pages)


def test_crawl_yields_all_completed_pages_when_workers_finish_ahead() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            text="<html><title>Page</title><body>content</body></html>",
            headers={"content-type": "text/html"},
        )

    async def run() -> list[CrawlPage]:
        seeds = [f"https://example.test/page-{number}" for number in range(30)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler = AsyncCrawler(
                client,
                user_agent="findex-test",
                concurrency=20,
                per_host=20,
                host_delay=0,
            )
            return [page async for page in crawler.crawl(seeds, max_pages=30)]

    pages = asyncio.run(run())
    assert len(pages) == 30
    assert len({page.url for page in pages}) == 30


def test_public_fetch_module_api_uses_client_user_agent() -> None:
    from findex.crawler.fetch import fetch

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": "findex-test (+https://example.test/contact)"},
        ) as client:
            result = await fetch(client, "https://example.test/")
        assert result.status == 200
        assert result.attempts == 1

    asyncio.run(run())
