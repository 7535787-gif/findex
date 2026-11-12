"""Command-line interface for findex."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .index import build_index, save_index
from .index_model import Index
from .ranking import BM25, Scorer, TfIdf
from .ranking import search as ranked_search
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
    corpus: Annotated[Path, typer.Argument(help="Directory containing .txt files")],
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
    if not corpus.is_dir():
        raise typer.BadParameter(f"Corpus directory does not exist: {corpus}")
    try:
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
    except OSError, ValueError:
        logger.exception("Index build failed")
        raise


@app.command("search")
def search_command(
    index_file: Annotated[Path, typer.Argument(help="Path to index.bin")],
    query: Annotated[str, typer.Argument(help="Search terms")],
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking algorithm")
    ] = "bm25",
    top_k: Annotated[
        int,
        typer.Option("--k", "--top-k", min=1, max=1000, help="Number of results"),
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", help="Write machine-readable JSON to stdout")
    ] = False,
) -> None:
    """Search the index and rank matching documents."""
    try:
        index = _load(index_file)
        scorer_impl: Scorer = BM25() if scorer == "bm25" else TfIdf()
        terms = list(tokenize(query))
        if not terms:
            raise ValueError("Query contains no searchable terms")
        results = ranked_search(index, terms, scorer_impl, top_k=top_k)
        if json_output:
            for item in results:
                payload = {
                    "doc_id": item.doc_id,
                    "score": item.score,
                    "title": item.title,
                    "path": item.path,
                    "snippet": item.snippet,
                }
                typer.echo(json.dumps(payload, ensure_ascii=False))
            return
        table = Table(title=f"Search results: {query}")
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Document")
        table.add_column("Path")
        table.add_column("Snippet", overflow="fold")
        for rank, item in enumerate(results, start=1):
            table.add_row(
                str(rank),
                f"{item.score:.4f}",
                item.title,
                item.path,
                item.snippet,
            )
        console.print(table)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
    except (OSError, ValueError, KeyError) as exc:
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


if __name__ == "__main__":
    app()
