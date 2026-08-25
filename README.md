# Multi-threaded Web Crawler + TF-IDF Search Engine

A small search engine built from scratch: a thread-pool web crawler, an inverted index
with TF-IDF weights, and a FastAPI search endpoint. No search libraries — only
`requests`, `BeautifulSoup4`, `FastAPI`, `uvicorn`, `typer` and the standard library
(SQLite included).

## Architecture

```
seed URLs ──▶ crawler.py ──▶ documents ──▶ indexer.py ──▶ terms + postings ──▶ api.py
              (threads)      (SQLite)      (TF-IDF)        (inverted index)     (/search)
```

| Module | Responsibility |
| --- | --- |
| `config.py` | Loads `~/.secrets/search_engine_config.json`, validates it, applies env overrides |
| `storage.py` | SQLite schema and connection helpers shared by the other modules |
| `crawler.py` | Multi-threaded BFS crawler; extracts text and links, writes `documents` |
| `indexer.py` | Tokenizes documents, computes TF/IDF/TF-IDF, writes `terms` and `postings` |
| `api.py` | FastAPI app: `/search`, `/stats`, `/health` |
| `cli.py` | `typer` CLI: `crawl`, `index`, `serve`, `stats` |

### Database schema

| Table | Columns |
| --- | --- |
| `documents` | `id`, `url` (unique), `title`, `text`, `fetched_at`, `term_count` |
| `terms` | `term` (PK), `df`, `idf` |
| `postings` | `term`, `doc_id`, `tf`, `tfidf` — the inverted index |
| `index_meta` | `key`, `value` — e.g. when the index was last built |

## Configuration

Create `~/.secrets/search_engine_config.json` (never committed):

```json
{
  "seed_urls": ["https://docs.python.org/3/library/unittest.html"],
  "max_pages": 25,
  "db_path": "~/.local/share/search_engine/search.db",
  "max_workers": 8,
  "request_timeout": 10,
  "same_domain_only": true,
  "user_agent": "MiniSearchBot/1.0"
}
```

`seed_urls`, `max_pages` and `db_path` are required; the rest are optional. Every value
can be overridden with an environment variable — `SEARCH_ENGINE_CONFIG`,
`SEARCH_ENGINE_DB_PATH`, `SEARCH_ENGINE_MAX_PAGES`, `SEARCH_ENGINE_SEED_URLS`,
`SEARCH_ENGINE_MAX_WORKERS` — which is how the container points the database at `/data`.

## Running locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/python cli.py crawl --max-pages 25
```

```bash
.venv/bin/python cli.py serve --port 8000
```

`crawl` rebuilds the index when it finishes; pass `--skip-index` to skip that, or run
`cli.py index` on its own.

## Search API

```bash
curl "http://localhost:8000/search?q=test"
```

```json
{
  "query": "test",
  "terms": ["test"],
  "count": 3,
  "results": [
    {
      "url": "https://docs.python.org/3/library/unittest.html",
      "title": "unittest — Unit testing framework",
      "score": 0.059879,
      "matched_terms": ["test"],
      "snippet": "…Test Discovery Organizing test code…"
    }
  ]
}
```

`GET /search?q=<query>&limit=<n>` — `limit` defaults to 5. Also available: `/stats`,
`/health`, and interactive docs at `/docs`.

## How ranking works

For a document `d` and term `t`:

```
tf(t, d)    = count(t in d) / total_tokens(d)
df(t)       = number of documents containing t
idf(t)      = ln((1 + N) / (1 + df(t))) + 1
tfidf(t, d) = tf(t, d) * idf(t)
```

The IDF is smoothed the way scikit-learn smooths it: the `+1` terms avoid dividing by
zero, and the trailing `+1` stops a term that appears in every document from collapsing
to a weight of exactly zero. A document's score for a query is the sum of the TF-IDF
weights of the query terms it contains; the top `limit` documents are returned.

Tokenization is deliberately simple — lowercase, split on runs of alphanumeric
characters, drop punctuation. There is no stemming, so `cat` and `cats` are distinct
terms.

## Concurrency model

`crawler.py` keeps a `ThreadPoolExecutor` saturated from a shared frontier:

- a `threading.Lock` guards the visited set and the remaining page budget, so a URL is
  claimed exactly once and `max_pages` is never exceeded;
- a second lock serialises SQLite writes;
- each worker thread gets its own `requests.Session` (Sessions are not thread-safe);
- the database runs in WAL mode so the API can read while a crawl is writing;
- a failed fetch returns its page budget so the crawl still reaches `max_pages`.

## Docker

```bash
docker-compose build
```

Run a crawl (writes to the `search-data` volume mounted at `/data`):

```bash
docker-compose run --rm search-engine crawl
```

Start the API (the default command):

```bash
docker-compose up -d
```

```bash
curl "http://localhost:8000/search?q=test"
```

The compose file mounts `~/.secrets` read-only into the container so the config is never
baked into the image, and keeps the SQLite database on a named volume at `/data`.

## Tests

```bash
.venv/bin/python -m pytest
```

128 tests, no network access — every `requests.get` is mocked. They cover URL
normalization and link extraction, text extraction, the crawler's thread-safety
invariants (page budget, no duplicate URLs, identical results at 1 and 8 workers), the
tokenizer, the TF/IDF/TF-IDF maths against hand-computed values, ranking order, and the
API endpoints.
