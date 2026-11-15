"""Polite asyncio crawler with robots.txt, bounded concurrency and JSONL-ready pages."""

from __future__ import annotations

import asyncio
import email.utils
import html
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import TextIO
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger("findex.crawl")


def canonicalize_url(url: str) -> str:
    """Normalize scheme/host, fragments, trailing slash and query ordering."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host or scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL: {url}")
    port = parts.port
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    if port is not None and not default_port:
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(str(attrs_dict["href"]))
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip_depth:
            cleaned = " ".join(data.split())
            if cleaned:
                self.text_parts.append(cleaned)


def _parse_html(body: bytes, url: str) -> tuple[str, str, list[str]]:
    parser = _PageParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    title = " ".join(" ".join(parser.title_parts).split()) or url
    text = html.unescape(" ".join(parser.text_parts))
    return title, text, parser.links


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    requests_in_flight: int = 0
    queue_depth: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_second(self) -> float:
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return self.pages / elapsed


@dataclass(frozen=True, slots=True)
class CrawlPage:
    url: str
    title: str
    text: str
    status: int
    byte_count: int
    elapsed_s: float
    depth: int
    fetched_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# Public domain name used in the laboratory specification.
Page = CrawlPage


def _empty_headers() -> dict[str, str]:
    return {}


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    body: bytes
    elapsed_s: float
    attempts: int
    error: str | None = None
    headers: dict[str, str] = field(default_factory=_empty_headers)
    final_url: str | None = None


class RequestGate:
    """Applies global/per-host limits and counts active HTTP requests."""

    def __init__(
        self,
        concurrency: int,
        per_host: int,
        host_delay: float,
        on_request: Callable[[bool], None] | None = None,
    ) -> None:
        self.global_sem = asyncio.Semaphore(concurrency)
        self.per_host = per_host
        self.host_delay = max(0.0, host_delay)
        self.host_sems: dict[str, asyncio.Semaphore] = {}
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.last_started: dict[str, float] = {}
        self.on_request = on_request

    async def get(
        self, client: httpx.AsyncClient, url: str, timeout: float
    ) -> httpx.Response:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        host_sem = self.host_sems.setdefault(host, asyncio.Semaphore(self.per_host))
        host_lock = self.host_locks.setdefault(host, asyncio.Lock())
        async with host_sem, self.global_sem:
            async with host_lock:
                now = time.monotonic()
                wait = self.host_delay - (now - self.last_started.get(host, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
                self.last_started[host] = time.monotonic()
            if self.on_request:
                self.on_request(True)
            try:
                return await client.get(url, timeout=timeout)
            finally:
                if self.on_request:
                    self.on_request(False)


class RobotsCache:
    """Fetches and caches one robots.txt policy per origin."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        gate: RequestGate,
        user_agent: str,
        timeout: float,
    ) -> None:
        self.client = client
        self.gate = gate
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                if origin not in self._cache:
                    parser = RobotFileParser()
                    robots_url = f"{origin}/robots.txt"
                    try:
                        response = await self.gate.get(
                            self.client, robots_url, self.timeout
                        )
                        if response.status_code == 404:
                            lines: list[str] = []
                        elif response.is_success:
                            lines = response.text.splitlines()
                        else:
                            # Fail closed if the robots policy cannot be read.
                            lines = ["User-agent: *", "Disallow: /"]
                    except (httpx.HTTPError, TimeoutError):
                        lines = ["User-agent: *", "Disallow: /"]
                    parser.parse(lines)
                    self._cache[origin] = parser
        return self._cache[origin].can_fetch(self.user_agent, url)


class AsyncCrawler:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str,
        concurrency: int = 5,
        per_host: int = 2,
        host_delay: float = 0.25,
        timeout: float = 10.0,
        max_retries: int = 3,
        max_redirects: int = 10,
        stats: CrawlStats | None = None,
        on_stats: Callable[[CrawlStats], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
        allowed_domains: set[str] | None = None,
    ) -> None:
        if concurrency < 1 or per_host < 1:
            raise ValueError("Concurrency and per-host limit must be positive")
        self.client = client
        normalized_domains = (
            {
                domain.strip().lower().rstrip(".").removeprefix("www.")
                for domain in allowed_domains
                if domain.strip()
            }
            if allowed_domains is not None
            else None
        )
        self.allowed_domains = normalized_domains
        self.worker_count = concurrency
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.max_redirects = max(0, max_redirects)
        self.stats = stats or CrawlStats()
        self.on_stats = on_stats
        self.sleep = sleep
        self.random_value = random_value
        self.gate = RequestGate(
            concurrency, per_host, host_delay, on_request=self._request_change
        )
        self.robots = RobotsCache(client, self.gate, user_agent, timeout)
        self.client.headers.update({"User-Agent": user_agent})

    def _is_allowed_domain(self, url: str) -> bool:
        """Return whether URL belongs to one of the configured domains."""
        if self.allowed_domains is None:
            return True

        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        host = host.removeprefix("www.")
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.allowed_domains
        )

    def _notify(self) -> None:
        if self.on_stats:
            self.on_stats(self.stats)

    def _request_change(self, started: bool) -> None:
        if started:
            self.stats.requests_in_flight += 1
        else:
            self.stats.requests_in_flight = max(0, self.stats.requests_in_flight - 1)
        self._notify()

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    async def fetch(self, url: str) -> FetchResult:
        started = time.perf_counter()
        last_status: int | None = None
        last_body = b""
        last_headers: dict[str, str] = {}
        last_error: str | None = None
        final_url = url

        attempt = 0
        for attempt in range(1, self.max_retries + 2):
            last_status = None
            last_body = b""
            last_headers = {}
            final_url = url
            visited = {canonicalize_url(url)}
            redirects = 0
            retry_delay: float | None = None
            try:
                while True:
                    async with asyncio.timeout(self.timeout):
                        response = await self.gate.get(
                            self.client, final_url, self.timeout
                        )
                    last_status = response.status_code
                    last_body = response.content
                    last_headers = {
                        str(key): str(value) for key, value in response.headers.items()
                    }

                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            last_error = f"HTTP {response.status_code} without Location"
                            break
                        if redirects >= self.max_redirects:
                            last_error = "redirect loop: maximum redirects exceeded"
                            break
                        try:
                            next_url = canonicalize_url(urljoin(final_url, location))
                        except ValueError:
                            last_error = "redirect target is not an HTTP(S) URL"
                            break
                        if next_url in visited:
                            last_error = "redirect loop: URL visited more than once"
                            break
                        if not self._is_allowed_domain(next_url):
                            last_error = "redirect target domain is not allowed"
                            break
                        if not await self.robots.allowed(next_url):
                            last_error = "redirect target disallowed by robots.txt"
                            break
                        visited.add(next_url)
                        final_url = next_url
                        redirects += 1
                        continue

                    retryable = response.status_code == 429 or (
                        500 <= response.status_code <= 599
                    )
                    if retryable and attempt <= self.max_retries:
                        retry_delay = self._retry_after(
                            response.headers.get("Retry-After")
                        )
                        if retry_delay is None:
                            retry_delay = min(30.0, 0.5 * (2 ** (attempt - 1)))
                            retry_delay += self.random_value() * 0.25
                        break
                    if retryable:
                        last_error = (
                            f"HTTP {response.status_code} after {attempt} attempts"
                        )
                    elif response.status_code >= 400:
                        last_error = f"HTTP {response.status_code}"
                    else:
                        last_error = None
                    break

                if retry_delay is not None:
                    await self.sleep(retry_delay)
                    continue
                break
            except (TimeoutError, httpx.TimeoutException) as exc:
                last_error = f"timeout: {type(exc).__name__}"
                if attempt <= self.max_retries:
                    delay = min(30.0, 0.5 * (2 ** (attempt - 1)))
                    delay += self.random_value() * 0.25
                    await self.sleep(delay)
                else:
                    break
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break

        elapsed = time.perf_counter() - started
        result = FetchResult(
            url,
            last_status,
            last_body,
            elapsed,
            attempt,
            last_error,
            last_headers,
            final_url,
        )
        logger.info(
            "url=%s final_url=%s status=%s bytes=%d elapsed=%.3f attempts=%d error=%s",
            url,
            final_url,
            last_status if last_status is not None else "ERR",
            len(last_body),
            elapsed,
            attempt,
            last_error or "-",
        )
        if last_error:
            self.stats.errors += 1
            self._notify()
        return result

    async def crawl(
        self,
        start_url: str | Iterable[str],
        *,
        max_pages: int,
        max_depth: int = 3,
    ) -> AsyncIterator[CrawlPage]:
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        raw_seeds = [start_url] if isinstance(start_url, str) else list(start_url)
        seeds: list[str] = []
        for raw_seed in raw_seeds:
            seed = canonicalize_url(raw_seed)
            if not self._is_allowed_domain(seed):
                raise ValueError(f"Seed URL domain is not allowed: {seed}")
            if seed not in seeds and len(seeds) < max_pages:
                seeds.append(seed)
        if not seeds:
            raise ValueError("At least one seed URL is required")
        queue: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
        output: asyncio.Queue[CrawlPage | None] = asyncio.Queue()
        seen = set(seeds)
        pending = len(seeds)
        pending_lock = asyncio.Lock()
        for seed in seeds:
            await queue.put((seed, 0))

        async def worker() -> None:
            nonlocal pending
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    url, depth = item
                    self.stats.queue_depth = queue.qsize()
                    self._notify()
                    if not await self.robots.allowed(url):
                        logger.info(
                            "url=%s status=ROBOTS_DISALLOW "
                            "bytes=0 elapsed=0 attempts=0",
                            url,
                        )
                    else:
                        result = await self.fetch(url)
                        if (
                            result.error is None
                            and result.status is not None
                            and 200 <= result.status < 300
                        ):
                            content_type = result.headers.get(
                                "content-type", ""
                            ).lower()
                            if "html" in content_type or not content_type:
                                page_url = result.final_url or url
                                try:
                                    title, text, links = await asyncio.to_thread(
                                        _parse_html, result.body, page_url
                                    )
                                except (
                                    UnicodeError,
                                    ValueError,
                                ) as exc:  # malformed HTML
                                    logger.warning(
                                        "HTML parse failed for %s: %s", url, exc
                                    )
                                    title, text, links = url, "", []
                                page = CrawlPage(
                                    page_url,
                                    title,
                                    text,
                                    result.status,
                                    len(result.body),
                                    result.elapsed_s,
                                    depth,
                                )
                                await output.put(page)
                                self.stats.pages += 1
                                self._notify()
                                if depth < max_depth:
                                    for href in links:
                                        try:
                                            child = canonicalize_url(
                                                urljoin(page_url, href)
                                            )
                                        except ValueError:
                                            continue
                                        async with pending_lock:
                                            if (
                                                self._is_allowed_domain(child)
                                                and child not in seen
                                                and len(seen) < max_pages
                                            ):
                                                seen.add(child)
                                                pending += 1
                                                await queue.put((child, depth + 1))
                    async with pending_lock:
                        pending -= 1
                        if pending == 0:
                            await output.put(None)
                    self.stats.queue_depth = queue.qsize()
                    self._notify()
                finally:
                    queue.task_done()

        worker_count = self.worker_count
        tasks: list[asyncio.Task[None]] = []
        async with asyncio.TaskGroup() as group:
            for _ in range(worker_count):
                tasks.append(group.create_task(worker()))
            yielded_pages = 0
            reached_limit = False
            while True:
                page = await output.get()
                if page is None:
                    break
                yielded_pages += 1
                yield page
                if yielded_pages >= max_pages:
                    reached_limit = True
                    for task in tasks:
                        task.cancel()
                    break
            if not reached_limit:
                for _ in tasks:
                    await queue.put(None)


def write_jsonl_page(file_obj: TextIO, page: CrawlPage) -> None:
    """Write one page immediately; accepts a text file-like object."""
    record = {
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "status": page.status,
        "bytes": page.byte_count,
        "elapsed_s": round(page.elapsed_s, 6),
        "depth": page.depth,
        "fetched_at": page.fetched_at,
    }
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
