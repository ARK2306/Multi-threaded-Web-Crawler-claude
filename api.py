"""FastAPI search API backed by the SQLite inverted index.

``GET /search?q=...`` tokenizes the query the same way the indexer does, sums the
precomputed TF-IDF weights of the matching terms per document, and returns the
highest-scoring documents.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query

import storage
from config import Config, ConfigError, load_config
from indexer import tokenize

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
SNIPPET_RADIUS = 120


@dataclass
class SearchHit:
    url: str
    title: str
    score: float
    matched_terms: list[str]
    snippet: str

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "score": round(self.score, 6),
            "matched_terms": self.matched_terms,
            "snippet": self.snippet,
        }


def make_snippet(text: str, terms: list[str]) -> str:
    """Return a short window of ``text`` around the first matching term."""
    if not text:
        return ""
    lowered = text.lower()
    position = -1
    for term in terms:
        found = lowered.find(term)
        if found != -1 and (position == -1 or found < position):
            position = found
    if position == -1:
        return text[: SNIPPET_RADIUS * 2].strip()

    start = max(0, position - SNIPPET_RADIUS)
    end = min(len(text), position + SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    return f"{'…' if start > 0 else ''}{snippet}{'…' if end < len(text) else ''}"


def search(conn: sqlite3.Connection, query: str, limit: int = DEFAULT_LIMIT) -> list[SearchHit]:
    """Score every document containing any query term and return the best ``limit``."""
    terms = sorted(set(tokenize(query)))
    if not terms:
        return []

    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(
        f"""
        SELECT d.url        AS url,
               d.title      AS title,
               d.text       AS text,
               SUM(p.tfidf) AS score,
               GROUP_CONCAT(DISTINCT p.term) AS matched
        FROM postings p
        JOIN documents d ON d.id = p.doc_id
        WHERE p.term IN ({placeholders})
        GROUP BY p.doc_id
        ORDER BY score DESC, d.url ASC
        LIMIT ?
        """,
        (*terms, limit),
    ).fetchall()

    hits = []
    for row in rows:
        matched = sorted((row["matched"] or "").split(","))
        hits.append(
            SearchHit(
                url=row["url"],
                title=row["title"] or row["url"],
                score=float(row["score"]),
                matched_terms=matched,
                snippet=make_snippet(row["text"] or "", matched),
            )
        )
    return hits


def create_app(config: Config | None = None) -> FastAPI:
    """Build the FastAPI app. The config is loaded lazily so import never fails."""
    state: dict[str, object] = {"config": config, "conn": None, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            cfg = state["config"] or load_config()
            state["config"] = cfg
            state["conn"] = storage.init_db(cfg.db_path)
            logger.info("search API ready (db=%s)", cfg.db_path)
        except ConfigError as exc:
            # Keep the process up so /health can explain what is wrong.
            state["error"] = str(exc)
            logger.error("configuration error: %s", exc)
        yield
        conn = state.get("conn")
        if isinstance(conn, sqlite3.Connection):
            conn.close()

    app = FastAPI(
        title="Mini Search Engine",
        description="Multi-threaded crawler + inverted index + TF-IDF search.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_conn() -> sqlite3.Connection:
        conn = state.get("conn")
        if not isinstance(conn, sqlite3.Connection):
            raise HTTPException(status_code=503, detail=state.get("error") or "database unavailable")
        return conn

    @app.get("/")
    def root() -> dict:
        return {
            "service": "mini-search-engine",
            "endpoints": ["/search?q=<query>", "/stats", "/health", "/docs"],
        }

    @app.get("/health")
    def health() -> dict:
        conn = state.get("conn")
        if not isinstance(conn, sqlite3.Connection):
            return {"status": "degraded", "detail": state.get("error") or "database unavailable"}
        return {"status": "ok", "documents": storage.document_count(conn)}

    @app.get("/stats")
    def stats() -> dict:
        conn = get_conn()
        terms = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        return {
            "documents": storage.document_count(conn),
            "terms": int(terms),
            "postings": int(postings),
            "indexed_at": storage.get_meta(conn, "indexed_at"),
        }

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., min_length=1, description="Search query"),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=50, description="Number of results"),
    ) -> dict:
        conn = get_conn()
        terms = sorted(set(tokenize(q)))
        hits = search(conn, q, limit)
        return {
            "query": q,
            "terms": terms,
            "count": len(hits),
            "results": [hit.as_dict() for hit in hits],
        }

    return app


app = create_app()
