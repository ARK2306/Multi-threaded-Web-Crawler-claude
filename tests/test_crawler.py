"""Crawler tests. Every network call is mocked — the suite never touches the internet."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

import crawler
import storage
from config import Config

SAMPLE_HTML = """
<html>
  <head>
    <title>Sample Page</title>
    <style>body { color: red; }</style>
    <script>var tracking = "should not be indexed";</script>
  </head>
  <body>
    <h1>Hello World</h1>
    <p>This is a <b>test</b> page about crawling.</p>
    <a href="/about">About</a>
    <a href="/about/">About with slash</a>
    <a href="page2.html">Relative</a>
    <a href="https://example.com/absolute">Absolute</a>
    <a href="https://other.com/external">External</a>
    <a href="#section">Fragment only</a>
    <a href="mailto:someone@example.com">Mail</a>
    <a href="javascript:void(0)">JS</a>
    <a href="/manual.pdf">PDF</a>
    <a href="/img/logo.png">Image</a>
    <a href="/about#team">About with fragment</a>
  </body>
</html>
"""


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(
        self,
        text="",
        status_code=200,
        content_type="text/html; charset=utf-8",
        url="",
        apparent_encoding="utf-8",
    ):
        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.apparent_encoding = apparent_encoding
        self.encoding = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        seed_urls=["https://example.com/"],
        max_pages=5,
        db_path=tmp_path / "test.db",
        max_workers=4,
        request_timeout=1.0,
        same_domain_only=True,
    )


# -- URL normalization ------------------------------------------------------------------


class TestNormalizeUrl:
    def test_resolves_relative_paths(self):
        assert (
            crawler.normalize_url("page2.html", "https://example.com/docs/index.html")
            == "https://example.com/docs/page2.html"
        )

    def test_resolves_root_relative_paths(self):
        assert (
            crawler.normalize_url("/about", "https://example.com/docs/index.html")
            == "https://example.com/about"
        )

    def test_strips_fragment(self):
        assert (
            crawler.normalize_url("https://example.com/about#team")
            == "https://example.com/about"
        )

    def test_strips_trailing_slash_but_keeps_root(self):
        assert crawler.normalize_url("https://example.com/about/") == "https://example.com/about"
        assert crawler.normalize_url("https://example.com/") == "https://example.com"

    def test_lowercases_host_only(self):
        assert (
            crawler.normalize_url("https://EXAMPLE.com/CaseSensitive")
            == "https://example.com/CaseSensitive"
        )

    def test_drops_default_ports(self):
        assert crawler.normalize_url("https://example.com:443/a") == "https://example.com/a"
        assert crawler.normalize_url("http://example.com:80/a") == "http://example.com/a"

    def test_keeps_query_string(self):
        assert (
            crawler.normalize_url("https://example.com/search?q=test")
            == "https://example.com/search?q=test"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "mailto:a@b.com",
            "javascript:void(0)",
            "tel:+15551234",
            "#anchor",
            "",
            "   ",
            "ftp://example.com/file",
            "https://example.com/manual.pdf",
            "https://example.com/img/logo.png",
            "https://example.com/app.js",
        ],
    )
    def test_rejects_non_crawlable(self, url):
        assert crawler.normalize_url(url) is None


# -- Link extraction --------------------------------------------------------------------


class TestExtractLinks:
    @pytest.fixture
    def links(self):
        return crawler.extract_links(SAMPLE_HTML, "https://example.com/index.html")

    def test_finds_expected_links(self, links):
        assert "https://example.com/about" in links
        assert "https://example.com/page2.html" in links
        assert "https://example.com/absolute" in links
        assert "https://other.com/external" in links

    def test_excludes_non_crawlable_links(self, links):
        for rejected in (
            "mailto:someone@example.com",
            "javascript:void(0)",
            "https://example.com/manual.pdf",
            "https://example.com/img/logo.png",
        ):
            assert rejected not in links

    def test_deduplicates_after_normalization(self, links):
        # /about, /about/ and /about#team all collapse to a single URL.
        assert links.count("https://example.com/about") == 1

    def test_no_duplicates_at_all(self, links):
        assert len(links) == len(set(links))

    def test_empty_document(self):
        assert crawler.extract_links("<html><body>no links</body></html>", "https://example.com") == []

    def test_honours_base_href(self):
        html = '<html><head><base href="https://cdn.example.com/v2/"></head><body><a href="doc">D</a></body></html>'
        assert crawler.extract_links(html, "https://example.com/page") == [
            "https://cdn.example.com/v2/doc"
        ]


# -- Text extraction --------------------------------------------------------------------


class TestExtractText:
    def test_returns_title_and_visible_text(self):
        title, text = crawler.extract_text(SAMPLE_HTML)
        assert title == "Sample Page"
        assert "Hello World" in text
        assert "This is a test page about crawling." in text

    def test_drops_script_and_style_content(self):
        _, text = crawler.extract_text(SAMPLE_HTML)
        assert "should not be indexed" not in text
        assert "color: red" not in text

    def test_collapses_whitespace(self):
        _, text = crawler.extract_text("<p>a\n\n   b\t\tc</p>")
        assert text == "a b c"

    def test_missing_title(self):
        title, text = crawler.extract_text("<html><body>hi</body></html>")
        assert title == ""
        assert text == "hi"


# -- Domain policy ----------------------------------------------------------------------


class TestSameDomain:
    def test_matching_hosts(self):
        assert crawler.same_domain("https://example.com/a", "https://example.com/b")

    def test_www_is_ignored(self):
        assert crawler.same_domain("https://www.example.com/a", "https://example.com/b")

    def test_different_hosts(self):
        assert not crawler.same_domain("https://example.com/a", "https://other.com/b")

    def test_subdomains_are_different(self):
        assert not crawler.same_domain("https://docs.example.com/a", "https://example.com/b")


# -- Crawler behaviour (requests.get mocked) --------------------------------------------


class TestCrawlerFetch:
    def test_fetch_parses_page(self, config):
        crawl = crawler.Crawler(config)
        response = FakeResponse(SAMPLE_HTML, url="https://example.com/index.html")
        with patch.object(requests.Session, "get", return_value=response) as mocked:
            page = crawl.fetch("https://example.com/index.html")
        mocked.assert_called_once()
        assert page is not None
        assert page.title == "Sample Page"
        assert "crawling" in page.text
        assert "https://example.com/about" in page.links
        crawl.close()

    def test_fetch_returns_none_on_http_error(self, config):
        crawl = crawler.Crawler(config)
        with patch.object(requests.Session, "get", return_value=FakeResponse("", status_code=404)):
            assert crawl.fetch("https://example.com/missing") is None
        crawl.close()

    def test_fetch_returns_none_on_network_error(self, config):
        crawl = crawler.Crawler(config)
        with patch.object(requests.Session, "get", side_effect=requests.Timeout("timed out")):
            assert crawl.fetch("https://example.com/slow") is None
        crawl.close()

    def test_fetch_skips_non_html(self, config):
        crawl = crawler.Crawler(config)
        response = FakeResponse("{}", content_type="application/json", url="https://example.com/api")
        with patch.object(requests.Session, "get", return_value=response):
            assert crawl.fetch("https://example.com/api") is None
        crawl.close()

    def test_fetch_sniffs_encoding_when_header_omits_charset(self, config):
        """Without a charset header requests would decode UTF-8 as latin-1."""
        crawl = crawler.Crawler(config)
        response = FakeResponse(
            SAMPLE_HTML, content_type="text/html", url="https://example.com/x", apparent_encoding="utf-8"
        )
        with patch.object(requests.Session, "get", return_value=response):
            crawl.fetch("https://example.com/x")
        assert response.encoding == "utf-8"
        crawl.close()

    def test_fetch_trusts_charset_from_header(self, config):
        crawl = crawler.Crawler(config)
        response = FakeResponse(SAMPLE_HTML, url="https://example.com/x")
        with patch.object(requests.Session, "get", return_value=response):
            crawl.fetch("https://example.com/x")
        assert response.encoding is None  # left untouched
        crawl.close()

    def test_fetch_uses_final_redirect_url(self, config):
        crawl = crawler.Crawler(config)
        response = FakeResponse(SAMPLE_HTML, url="https://example.com/final/")
        with patch.object(requests.Session, "get", return_value=response):
            page = crawl.fetch("https://example.com/start")
        assert page.url == "https://example.com/final"
        crawl.close()


class TestCrawlerPolicy:
    def test_stays_on_seed_domain(self, config):
        crawl = crawler.Crawler(config)
        assert crawl.is_allowed("https://example.com/anything")
        assert not crawl.is_allowed("https://other.com/anything")
        crawl.close()

    def test_allows_any_domain_when_disabled(self, config):
        config.same_domain_only = False
        crawl = crawler.Crawler(config)
        assert crawl.is_allowed("https://other.com/anything")
        crawl.close()

    def test_allow_list_takes_precedence(self, config):
        config.allowed_domains = ["other.com"]
        crawl = crawler.Crawler(config)
        assert crawl.is_allowed("https://sub.other.com/page")
        assert not crawl.is_allowed("https://example.com/page")
        crawl.close()

    def test_claim_is_exclusive(self, config):
        crawl = crawler.Crawler(config)
        assert crawl._claim("https://example.com/a") is True
        assert crawl._claim("https://example.com/a") is False
        crawl.close()

    def test_claim_respects_budget(self, config):
        config.max_pages = 2
        crawl = crawler.Crawler(config)
        assert crawl._claim("https://example.com/1")
        assert crawl._claim("https://example.com/2")
        assert not crawl._claim("https://example.com/3")
        crawl.close()


class TestCrawlEndToEnd:
    """Drive a full multi-threaded crawl over a fake three-page site."""

    SITE = {
        "https://example.com": '<html><title>Home</title><body>home page '
        '<a href="/a">A</a> <a href="/b">B</a></body></html>',
        "https://example.com/a": '<html><title>A</title><body>page a '
        '<a href="/b">B</a> <a href="https://other.com/x">X</a></body></html>',
        "https://example.com/b": '<html><title>B</title><body>page b <a href="/">Home</a></body></html>',
    }

    def _fake_get(self, url, **_kwargs):
        key = crawler.normalize_url(url)
        if key not in self.SITE:
            return FakeResponse("", status_code=404, url=url)
        return FakeResponse(self.SITE[key], url=key)

    def test_crawls_whole_site_and_persists(self, config):
        with patch.object(requests.Session, "get", side_effect=self._fake_get):
            crawl = crawler.Crawler(config)
            stats = crawl.crawl()
            urls = {row["url"] for row in crawl.conn.execute("SELECT url FROM documents")}
            crawl.close()

        assert stats.pages_crawled == 3
        assert urls == set(self.SITE)

    def test_respects_max_pages(self, config):
        config.max_pages = 2
        with patch.object(requests.Session, "get", side_effect=self._fake_get):
            crawl = crawler.Crawler(config)
            stats = crawl.crawl()
            count = storage.document_count(crawl.conn)
            crawl.close()

        assert stats.pages_crawled == 2
        assert count == 2

    def test_never_stores_a_url_twice(self, config):
        with patch.object(requests.Session, "get", side_effect=self._fake_get):
            crawl = crawler.Crawler(config)
            crawl.crawl()
            rows = crawl.conn.execute(
                "SELECT url, COUNT(*) c FROM documents GROUP BY url HAVING c > 1"
            ).fetchall()
            crawl.close()
        assert rows == []

    def test_stays_off_external_domains(self, config):
        with patch.object(requests.Session, "get", side_effect=self._fake_get):
            crawl = crawler.Crawler(config)
            crawl.crawl()
            urls = [row["url"] for row in crawl.conn.execute("SELECT url FROM documents")]
            crawl.close()
        assert not any("other.com" in url for url in urls)

    def test_concurrent_workers_stay_consistent(self, config):
        """Same result with 8 threads as with 1 — no lost or duplicated pages."""
        config.max_workers = 8
        config.max_pages = 3
        with patch.object(requests.Session, "get", side_effect=self._fake_get):
            crawl = crawler.Crawler(config)
            stats = crawl.crawl()
            count = storage.document_count(crawl.conn)
            crawl.close()
        assert stats.pages_crawled == count == 3
