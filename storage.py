"""SQLite schema and connection helpers shared by the crawler, indexer and API.

Everything lives in a single SQLite file: crawled documents plus the inverted index
built from them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL DEFAULT '',
    fetched_at  TEXT NOT NULL,
    term_count  INTEGER NOT NULL DEFAULT 0
);

-- One row per distinct term: document frequency and the IDF derived from it.
CREATE TABLE IF NOT EXISTS terms (
    term  TEXT PRIMARY KEY,
    df    INTEGER NOT NULL,
    idf   REAL NOT NULL
);

-- The inverted index itself: term -> (document, tf, tf-idf weight).
CREATE TABLE IF NOT EXISTS postings (
    term    TEXT NOT NULL,
    doc_id  INTEGER NOT NULL,
    tf      REAL NOT NULL,
    tfidf   REAL NOT NULL,
    PRIMARY KEY (term, doc_id),
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id);

-- Free-form bookkeeping (e.g. when the index was last built).
CREATE TABLE IF NOT EXISTS index_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def connect(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with sane pragmas for concurrent readers and one writer."""
    path = Path(db_path).expanduser()
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection and make sure every table exists."""
    conn = connect(db_path)
    with conn:
        conn.executescript(SCHEMA)
    return conn


def document_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
