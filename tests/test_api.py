"""Search ranking and the FastAPI endpoints, running against an in-tmp SQLite index."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
import indexer
import storage
from config import Config

DOCUMENTS = [
    ("https://example.com/testing", "Testing Guide", "test test test unittest python testing guide"),
    ("https://example.com/python", "Python Docs", "python is a language, this page mentions test once"),
    ("https://example.com/cooking", "Cooking", "recipes for bread and soup with no relevant words"),
    ("https://example.com/sqlite", "SQLite", "sqlite database test fixtures and python bindings"),
    ("https://example.com/threads", "Threads", "threading concurrency python workers"),
    ("https://example.com/extra", "Extra", "test coverage matters a lot in python projects"),
]


@pytest.fixture
def conn(tmp_path):
    connection = storage.init_db(tmp_path / "api.db")
    with connection:
        connection.executemany(
            "INSERT INTO documents (url, title, text, fetched_at) VALUES (?, ?, ?, ?)",
            [(url, title, text, "2026-01-01T00:00:00+00:00") for url, title, text in DOCUMENTS],
        )
    indexer.build_index(connection)
    yield connection
    connection.close()


@pytest.fixture
def client(tmp_path, conn):
    config = Config(seed_urls=["https://example.com"], max_pages=10, db_path=tmp_path / "api.db")
    with TestClient(api.create_app(config)) as test_client:
        yield test_client


class TestSearchFunction:
    def test_finds_matching_documents(self, conn):
        hits = api.search(conn, "test")
        assert hits
        assert all("test" in hit.matched_terms for hit in hits)

    def test_ranks_by_tfidf_descending(self, conn):
        hits = api.search(conn, "test")
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_densest_document_ranks_first(self, conn):
        # "test" appears three times in the short testing guide.
        assert api.search(conn, "test")[0].url == "https://example.com/testing"

    def test_multi_term_query_sums_weights(self, conn):
        hits = {hit.url: hit.score for hit in api.search(conn, "python test", limit=10)}
        python_only = {hit.url: hit.score for hit in api.search(conn, "python", limit=10)}
        test_only = {hit.url: hit.score for hit in api.search(conn, "test", limit=10)}
        url = "https://example.com/sqlite"
        assert hits[url] == pytest.approx(python_only[url] + test_only[url])

    def test_excludes_documents_without_the_terms(self, conn):
        urls = [hit.url for hit in api.search(conn, "test", limit=10)]
        assert "https://example.com/cooking" not in urls

    def test_defaults_to_five_results(self, conn):
        assert len(api.search(conn, "test python")) == 5

    def test_limit_is_honoured(self, conn):
        assert len(api.search(conn, "test python", limit=2)) == 2

    def test_query_is_case_insensitive(self, conn):
        assert [h.url for h in api.search(conn, "TEST")] == [h.url for h in api.search(conn, "test")]

    def test_punctuation_in_query_is_ignored(self, conn):
        assert [h.url for h in api.search(conn, "test!!!")] == [h.url for h in api.search(conn, "test")]

    @pytest.mark.parametrize("query", ["", "   ", "!!!", "nonexistentterm"])
    def test_no_results(self, conn, query):
        assert api.search(conn, query) == []

    def test_hit_carries_a_snippet(self, conn):
        assert api.search(conn, "test")[0].snippet


class TestSnippet:
    def test_centres_on_the_match(self):
        text = "alpha " * 60 + "needle" + " omega" * 60
        snippet = api.make_snippet(text, ["needle"])
        assert "needle" in snippet

    def test_marks_truncation(self):
        snippet = api.make_snippet("x" * 500 + " needle " + "y" * 500, ["needle"])
        assert snippet.startswith("…") and snippet.endswith("…")

    def test_handles_missing_term(self):
        assert api.make_snippet("some text", ["absent"]) == "some text"

    def test_handles_empty_text(self):
        assert api.make_snippet("", ["a"]) == ""


class TestEndpoints:
    def test_root_lists_endpoints(self, client):
        assert "/search?q=<query>" in client.get("/").json()["endpoints"]

    def test_health_reports_document_count(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["documents"] == len(DOCUMENTS)

    def test_stats(self, client):
        payload = client.get("/stats").json()
        assert payload["documents"] == len(DOCUMENTS)
        assert payload["terms"] > 0 and payload["postings"] > 0
        assert payload["indexed_at"]

    def test_search_returns_ranked_results(self, client):
        payload = client.get("/search", params={"q": "test"}).json()
        assert payload["query"] == "test"
        assert payload["terms"] == ["test"]
        assert payload["count"] == len(payload["results"])
        assert payload["results"][0]["url"] == "https://example.com/testing"

    def test_search_caps_at_five_by_default(self, client):
        assert len(client.get("/search", params={"q": "test python"}).json()["results"]) == 5

    def test_search_limit_parameter(self, client):
        assert len(client.get("/search", params={"q": "test python", "limit": 3}).json()["results"]) == 3

    def test_search_result_shape(self, client):
        result = client.get("/search", params={"q": "test"}).json()["results"][0]
        assert set(result) == {"url", "title", "score", "matched_terms", "snippet"}
        assert isinstance(result["score"], float)

    def test_search_without_query_is_rejected(self, client):
        assert client.get("/search").status_code == 422

    def test_search_with_empty_query_is_rejected(self, client):
        assert client.get("/search", params={"q": ""}).status_code == 422

    def test_unknown_term_returns_empty_results(self, client):
        payload = client.get("/search", params={"q": "zzzznotaword"}).json()
        assert payload["count"] == 0 and payload["results"] == []

    def test_limit_out_of_range_is_rejected(self, client):
        assert client.get("/search", params={"q": "test", "limit": 0}).status_code == 422
        assert client.get("/search", params={"q": "test", "limit": 999}).status_code == 422
