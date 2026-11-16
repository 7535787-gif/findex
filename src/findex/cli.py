"""Command-line interface for findex."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .crawler import AsyncCrawler, CrawlStats, write_jsonl_page
from .index import build_index, save_index
from .index_model import Index
from .ranking import BM25, Scorer, TfIdf
from .ranking import search_numpy
from .semantic import DEFAULT_MODEL, embed_index, hybrid_search, semantic_search
from .tokenize import tokenize

app = typer.Typer(
    name="findex",
    help="Local full-text search engine.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()
logger = logging.getLogger("findex")


def configure_logging(verbose: int) -> None:
    level = (
        logging.WARNING
        if verbose == 0
        else logging.INFO
        if verbose == 1
        else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=__import__("sys").stderr,
        force=True,
    )


def _load(path: Path) -> Index:
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")
    from .index import load_index

    postings, documents, _ = load_index(path)
    return Index(postings, documents)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="Increase log verbosity (-v, -vv)."
        ),
    ] = 0,
) -> None:
    """Configure logging once for the CLI."""
    configure_logging(verbose)


@app.command("index")
def index_command(
    corpus: Annotated[
        Path, typer.Argument(help="Directory of .txt files or crawler JSONL file")
    ],
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Output index file")
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Store token positions for phrase search"),
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Maximum number of documents")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", help="Execution backend"),
    ] = "serial",
) -> None:
    """Build and save an inverted index."""
    is_jsonl = corpus.is_file() and corpus.suffix.lower() == ".jsonl"
    if not corpus.is_dir() and not is_jsonl:
        raise typer.BadParameter(
            f"Corpus directory or crawler .jsonl file does not exist: {corpus}"
        )
    try:
        if is_jsonl:
            from .corpus import iter_documents

            total = sum(1 for _ in iter_documents(corpus))
        else:
            total = sum(1 for _ in corpus.rglob("*.txt"))
        if limit is not None:
            total = min(total, limit)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=Console(stderr=True),
        ) as progress:
            task = progress.add_task("Building index", total=total)
            postings, documents = build_index(
                corpus,
                progress_callback=lambda completed: progress.update(
                    task, completed=completed
                ),
                limit=limit,
                include_positions=positions,
                workers=workers,
                executor=executor,
            )
        save_index(postings, documents, output)
        console.print(f"[green]Index saved:[/green] {output}")
        console.print(f"Documents: {len(documents)} | Terms: {len(postings)}")
        logger.info("Index build completed")
    except (OSError, ValueError):
        logger.exception("Index build failed")
        raise


@app.command("search")
def search_command(
    index_file: Annotated[Path, typer.Argument(help="Path to index.bin")],
    query: Annotated[str, typer.Argument(help="Search terms")],
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking algorithm")
    ] = "bm25",
    mode: Annotated[
        Literal["bm25", "tfidf", "semantic", "hybrid"] | None,
        typer.Option("--mode", help="Retrieval mode"),
    ] = None,
    embeddings: Annotated[
        Path, typer.Option("--embeddings", help="Path to embeddings .npy file")
    ] = Path("embeddings.npy"),
    model: Annotated[
        str, typer.Option("--model", help="Sentence-transformer model")
    ] = DEFAULT_MODEL,
    top_k: Annotated[
        int,
        typer.Option("--k", "--top-k", min=1, max=1000, help="Number of results"),
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", help="Write machine-readable JSON to stdout")
    ] = False,
) -> None:
    """Search using lexical, semantic, or hybrid retrieval."""
    try:
        index = _load(index_file)
        terms = list(tokenize(query))
        if not terms:
            raise ValueError("Query contains no searchable terms")
        selected_mode = mode or scorer
        if selected_mode == "semantic":
            results = semantic_search(
                index, query, embeddings, model_name=model, top_k=top_k
            )
        elif selected_mode == "hybrid":
            results = hybrid_search(
                index, query, embeddings, model_name=model, top_k=top_k
            )
        else:
            active_scorer: Scorer = BM25() if selected_mode == "bm25" else TfIdf()
            results = search_numpy(index, terms, active_scorer, top_k=top_k)
        if json_output:
            for item in results:
                payload = {
                    "doc_id": item.doc_id,
                    "score": item.score,
                    "title": item.title,
                    "path": item.path,
                    "snippet": item.snippet,
                    "match_type": item.match_type,
                }
                typer.echo(json.dumps(payload, ensure_ascii=False))
            return
        table = Table(title=f"Search results ({selected_mode}): {query}")
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Match")
        table.add_column("Document")
        table.add_column("Path")
        table.add_column("Snippet", overflow="fold")
        for rank, item in enumerate(results, start=1):
            table.add_row(
                str(rank),
                f"{item.score:.4f}",
                item.match_type,
                item.title,
                item.path,
                item.snippet,
            )
        console.print(table)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("embed")
def embed_command(
    index_file: Annotated[Path, typer.Argument(help="Path to index.bin")],
    output: Annotated[
        Path, typer.Option("--out", help="Output embeddings .npy file")
    ] = Path("embeddings.npy"),
    model: Annotated[
        str, typer.Option("--model", help="Sentence-transformer model")
    ] = DEFAULT_MODEL,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
) -> None:
    """Create normalized embeddings for document chunks."""
    try:
        index = _load(index_file)
        path, chunks = embed_index(
            index, output, model_name=model, batch_size=batch_size
        )
        console.print(f"[green]Embeddings saved:[/green] {path}")
        console.print(f"Chunks: {chunks}")
    except (OSError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("stats")
def stats_command(
    index_file: Annotated[Path, typer.Argument(help="Path to index.bin")],
) -> None:
    """Show index statistics."""
    try:
        index = _load(index_file)
        table = Table(title="Index statistics")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Documents", str(index.num_docs))
        table.add_row("Terms", str(len(index)))
        table.add_row("Average document length", f"{index.avg_doc_length:.2f}")
        table.add_row(
            "Total postings", str(sum(len(items) for items in index.values()))
        )
        console.print(table)
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("crawl")
def crawl_command(
    start_url: Annotated[
        str,
        typer.Argument(
            help="Starting URL; crawl only sites you are authorized to access"
        ),
    ],
    contact: Annotated[
        str,
        typer.Option(
            "--contact",
            help="Contact email included in the crawler User-Agent",
        ),
    ],
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", min=1, help="Maximum URLs scheduled for fetching"),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, help="Global concurrent requests"),
    ] = 5,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Concurrent requests per host"),
    ] = 2,
    delay: Annotated[
        float,
        typer.Option(
            "--delay", min=0.0, help="Minimum seconds between requests to the same host"
        ),
    ] = 0.25,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 10.0,
    max_depth: Annotated[int, typer.Option("--max-depth", min=0)] = 5,
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Streaming JSONL output file")
    ] = Path("crawl.jsonl"),
    asyncio_debug: Annotated[
        bool, typer.Option("--asyncio-debug", help="Enable asyncio debug mode")
    ] = False,
) -> None:
    """Crawl an authorized website politely and stream pages to JSONL."""
    if "@" not in contact or contact.startswith("@") or contact.endswith("@"):
        raise typer.BadParameter("--contact must be a valid contact email address")
    user_agent = f"findex-lab6/1.0 (+mailto:{contact})"
    crawl_logger = logging.getLogger("findex.crawl")
    file_handler = logging.FileHandler("crawl.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = QueueHandler(log_queue)
    log_listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    crawl_logger.addHandler(queue_handler)
    crawl_logger.setLevel(logging.INFO)
    log_listener.start()
    stats = CrawlStats()

    def render_stats(current: CrawlStats) -> Table:
        table = Table(title="findex async crawler")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Pages collected", str(current.pages))
        table.add_row("Pages / second", f"{current.pages_per_second:.2f}")
        table.add_row("Requests in flight", str(current.requests_in_flight))
        table.add_row("Errors", str(current.errors))
        table.add_row("Queue depth", str(current.queue_depth))
        return table

    async def run(live: Live) -> None:
        transport = httpx.AsyncHTTPTransport(retries=0)
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            max_redirects=10,
            headers={"Accept": "text/html,application/xhtml+xml"},
        ) as client:
            start_host = (urlsplit(start_url).hostname or "").lower()
            if not start_host:
                raise ValueError("Start URL must contain a valid hostname")
            allowed_domains = {start_host.removeprefix("www.")}

            crawler = AsyncCrawler(
                client,
                user_agent=user_agent,
                concurrency=concurrency,
                per_host=per_host,
                host_delay=delay,
                timeout=timeout,
                stats=stats,
                on_stats=lambda current: live.update(render_stats(current)),
                allowed_domains=allowed_domains,
            )
            stream = await asyncio.to_thread(
                output.open, "w", encoding="utf-8", newline="\n"
            )
            try:
                async for page in crawler.crawl(
                    start_url, max_pages=max_pages, max_depth=max_depth
                ):
                    await asyncio.to_thread(write_jsonl_page, stream, page)
                    await asyncio.to_thread(stream.flush)
            finally:
                await asyncio.to_thread(stream.close)

    try:
        with Live(render_stats(stats), console=console, refresh_per_second=4) as live:
            asyncio.run(run(live), debug=asyncio_debug)
        console.print(f"[green]Crawl saved:[/green] {output}")
        console.print(f"Pages: {stats.pages} | Errors: {stats.errors} | Log: crawl.log")
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        crawl_logger.removeHandler(queue_handler)
        log_listener.stop()
        file_handler.close()


@app.command("serve")
def serve_command(
    port: Annotated[
        int | None, typer.Option("--port", min=1, max=65535, help="HTTP port")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Uvicorn worker processes")
    ] = 1,
) -> None:
    """Run the FastAPI search service with Uvicorn."""
    import uvicorn

    from .web.config import Settings

    settings = Settings()
    selected_port = port if port is not None else settings.port
    logger.info(
        "Starting findex web service on 0.0.0.0:%d with %d worker(s)",
        selected_port,
        workers,
    )
    uvicorn.run(
        "findex.web:app",
        host="0.0.0.0",
        port=selected_port,
        workers=workers,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    app()
