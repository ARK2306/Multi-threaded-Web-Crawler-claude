"""Inverted index builder.

Reads every crawled document out of SQLite, tokenizes it, and writes back an inverted
index containing, for each (term, document) pair, the term frequency and its TF-IDF
weight — plus the document frequency and IDF for each term.

Definitions used here
---------------------
``tf(t, d)``   = count(t in d) / total_tokens(d)          (length-normalised)
``df(t)``      = number of documents containing ``t``
``idf(t)``     = ln((1 + N) / (1 + df(t))) + 1            (smoothed, always > 0)
``tfidf(t, d)``= tf(t, d) * idf(t)

The IDF is smoothed the same way scikit-learn smooths it: the ``+1`` terms avoid a
division by zero for unseen terms, and the trailing ``+1`` keeps a term that appears in
every document from collapsing to a weight of exactly zero.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import storage

logger = logging.getLogger(__name__)

# Alphanumeric runs, underscore excluded: punctuation simply falls away.
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Tokens longer than this are almost always minified junk rather than words.
MAX_TOKEN_LENGTH = 40


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens, dropping punctuation."""
    if not text:
        return []
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) <= MAX_TOKEN_LENGTH
    ]


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """Length-normalised term frequencies: ``count(term) / len(tokens)``."""
    if not tokens:
        return {}
    total = len(tokens)
    return {term: count / total for term, count in Counter(tokens).items()}


def inverse_document_frequency(doc_count: int, doc_frequency: int) -> float:
    """Smoothed IDF: ``ln((1 + N) / (1 + df)) + 1``."""
    return math.log((1 + doc_count) / (1 + doc_frequency)) + 1.0


def compute_tfidf(documents: dict[int, str]) -> tuple[dict[str, float], dict[str, dict[int, tuple[float, float]]]]:
    """Build an in-memory inverted index from ``{doc_id: text}``.

    Returns ``(idf_by_term, postings)`` where ``postings[term][doc_id] == (tf, tfidf)``.
    """
    tf_by_doc: dict[int, dict[str, float]] = {}
    doc_frequency: Counter[str] = Counter()

    for doc_id, text in documents.items():
        tokens = tokenize(text)
        tfs = term_frequencies(tokens)
        tf_by_doc[doc_id] = tfs
        doc_frequency.update(tfs.keys())

    doc_count = len(documents)
    idf_by_term = {
        term: inverse_document_frequency(doc_count, df) for term, df in doc_frequency.items()
    }

    postings: dict[str, dict[int, tuple[float, float]]] = {}
    for doc_id, tfs in tf_by_doc.items():
        for term, tf in tfs.items():
            postings.setdefault(term, {})[doc_id] = (tf, tf * idf_by_term[term])

    return idf_by_term, postings


@dataclass
class IndexStats:
    documents: int = 0
    terms: int = 0
    postings: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"documents": self.documents, "terms": self.terms, "postings": self.postings}


def build_index(conn: sqlite3.Connection) -> IndexStats:
    """Rebuild the inverted index from scratch for every document in the database."""
    rows = conn.execute("SELECT id, text FROM documents").fetchall()
    documents = {int(row["id"]): row["text"] or "" for row in rows}

    if not documents:
        logger.warning("no documents to index — run the crawler first")
        with conn:
            conn.execute("DELETE FROM postings")
            conn.execute("DELETE FROM terms")
        return IndexStats()

    token_counts = {doc_id: len(tokenize(text)) for doc_id, text in documents.items()}
    idf_by_term, postings = compute_tfidf(documents)

    doc_frequency = {term: len(docs) for term, docs in postings.items()}
    posting_rows = [
        (term, doc_id, tf, tfidf)
        for term, docs in postings.items()
        for doc_id, (tf, tfidf) in docs.items()
    ]

    # The rebuild is one transaction, so readers (the API) never see a half-built index.
    with conn:
        conn.execute("DELETE FROM postings")
        conn.execute("DELETE FROM terms")
        conn.executemany(
            "UPDATE documents SET term_count = ? WHERE id = ?",
            [(count, doc_id) for doc_id, count in token_counts.items()],
        )
        conn.executemany(
            "INSERT INTO terms (term, df, idf) VALUES (?, ?, ?)",
            [(term, doc_frequency[term], idf) for term, idf in idf_by_term.items()],
        )
        conn.executemany(
            "INSERT INTO postings (term, doc_id, tf, tfidf) VALUES (?, ?, ?, ?)",
            posting_rows,
        )

    storage.set_meta(conn, "indexed_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    storage.set_meta(conn, "document_count", str(len(documents)))

    stats = IndexStats(documents=len(documents), terms=len(idf_by_term), postings=len(posting_rows))
    logger.info(
        "indexed %d documents, %d unique terms, %d postings",
        stats.documents,
        stats.terms,
        stats.postings,
    )
    return stats


def run_index(db_path: Path | str) -> IndexStats:
    """Convenience entry point used by the CLI."""
    conn = storage.init_db(db_path)
    try:
        return build_index(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from config import load_config

    print(run_index(load_config().db_path).as_dict())
