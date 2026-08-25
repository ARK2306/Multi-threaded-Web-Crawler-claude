"""Command line interface: ``crawl`` to populate the database, ``serve`` to search it."""

from __future__ import annotations

import logging
import sys

import typer

import indexer
import storage
from config import ConfigError, load_config

app = typer.Typer(
    add_completion=False,
    help="Multi-threaded web crawler with a TF-IDF search API.",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(config_path: str | None):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def crawl(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to the JSON config file."),
    max_pages: int = typer.Option(None, "--max-pages", "-n", help="Override max_pages."),
    workers: int = typer.Option(None, "--workers", "-w", help="Override the thread count."),
    skip_index: bool = typer.Option(False, "--skip-index", help="Crawl without rebuilding the index."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Crawl the seed URLs into SQLite, then rebuild the inverted index."""
    _setup_logging(verbose)
    config = _load(config_path)
    if max_pages is not None:
        config.max_pages = max_pages
    if workers is not None:
        config.max_workers = max(1, workers)

    typer.secho(
        f"Crawling {len(config.seed_urls)} seed URL(s) with {config.max_workers} threads, "
        f"max {config.max_pages} pages -> {config.db_path}",
        fg=typer.colors.CYAN,
    )

    # Imported here so `cli.py --help` works without requests/bs4 installed.
    import crawler

    stats = crawler.run_crawl(config)
    typer.secho(
        f"Crawl finished: {stats.pages_crawled} pages stored, {stats.pages_failed} failed.",
        fg=typer.colors.GREEN,
    )

    if skip_index:
        typer.echo("Skipping index build (--skip-index).")
        return

    index_stats = indexer.run_index(config.db_path)
    typer.secho(
        f"Index built: {index_stats.documents} documents, {index_stats.terms} terms, "
        f"{index_stats.postings} postings.",
        fg=typer.colors.GREEN,
    )


@app.command()
def index(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to the JSON config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Rebuild the inverted index from documents already in the database."""
    _setup_logging(verbose)
    config = _load(config_path)
    stats = indexer.run_index(config.db_path)
    typer.secho(
        f"Index built: {stats.documents} documents, {stats.terms} terms, "
        f"{stats.postings} postings.",
        fg=typer.colors.GREEN,
    )


@app.command()
def serve(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to the JSON config file."),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the FastAPI search server with uvicorn."""
    _setup_logging(False)
    config = _load(config_path)  # fail fast on a bad config instead of inside the server
    typer.secho(
        f"Serving http://{host}:{port}/search?q=... (db={config.db_path})",
        fg=typer.colors.CYAN,
    )

    import uvicorn

    uvicorn.run("api:app", host=host, port=port, reload=reload, log_level="info")


@app.command()
def stats(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to the JSON config file."),
) -> None:
    """Show what is currently stored in the database."""
    config = _load(config_path)
    conn = storage.init_db(config.db_path)
    try:
        documents = storage.document_count(conn)
        terms = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        indexed_at = storage.get_meta(conn, "indexed_at") or "never"
    finally:
        conn.close()
    typer.echo(f"database:  {config.db_path}")
    typer.echo(f"documents: {documents}")
    typer.echo(f"terms:     {terms}")
    typer.echo(f"postings:  {postings}")
    typer.echo(f"indexed:   {indexed_at}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        typer.secho("Interrupted.", fg=typer.colors.YELLOW, err=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
