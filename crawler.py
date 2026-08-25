"""Multi-threaded web crawler.

A pool of worker threads fetches pages concurrently while the main thread keeps the
pool saturated from a shared frontier. All mutable state (the visited set, the page
budget, the SQLite connection) is guarded by locks, so the crawl is safe to run with
any number of workers.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import storage
from config import Config

logger = logging.getLogger(__name__)

# Tags whose text is markup noise rather than page content.
NON_CONTENT_TAGS = ("script", "style", "noscript", "template", "svg", "head")

# Extensions we never want to download and try to parse as HTML.
SKIPPED_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".webm", ".ogg", ".wav",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
    ".exe", ".dmg", ".iso", ".deb", ".rpm", ".woff", ".woff2", ".ttf",
)


@dataclass
class Page:
    """A successfully fetched and parsed page."""

    url: str
    title: str
    text: str
    links: list[str]


@dataclass
class CrawlStats:
    pages_crawled: int = 0
    pages_failed: int = 0
    urls_seen: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "pages_crawled": self.pages_crawled,
            "pages_failed": self.pages_failed,
            "urls_seen": self.urls_seen,
        }


# --------------------------------------------------------------------------------------
# Pure helpers (no network, no database) — these are what the unit tests exercise.
# --------------------------------------------------------------------------------------


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    """Resolve ``url`` against ``base_url`` and canonicalise it.

    Returns ``None`` for anything that is not a crawlable http(s) page: mailto/javascript
    links, fragment-only links, and URLs pointing at binary assets.
    """
    if not url:
        return None
    candidate = url.strip()
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered.startswith(("mailto:", "javascript:", "tel:", "data:", "#")):
        return None

    if base_url:
        candidate = urljoin(base_url, candidate)

    candidate, _ = urldefrag(candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    path = parsed.path.lower()
    if path.endswith(SKIPPED_EXTENSIONS):
        return None

    # Drop trailing slashes so "/a", "/a/" and the two spellings of the site root
    # ("https://x" and "https://x/") each collapse to a single canonical URL.
    normalized_path = parsed.path.rstrip("/")

    netloc = parsed.netloc.lower()
    if (parsed.scheme == "http" and netloc.endswith(":80")) or (
        parsed.scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    rebuilt = f"{parsed.scheme}://{netloc}{normalized_path}"
    if parsed.query:
        rebuilt += f"?{parsed.query}"
    return rebuilt


def extract_links(html: str, base_url: str) -> list[str]:
    """Return the de-duplicated, absolute, crawlable links found in ``html``."""
    soup = BeautifulSoup(html, "html.parser")

    # A <base href> overrides the document URL for relative link resolution. It is
    # resolved but deliberately not normalized: stripping its trailing slash would
    # make every relative link resolve one path segment too high.
    base_tag = soup.find("base", href=True)
    if base_tag:
        resolved_base, _ = urldefrag(urljoin(base_url, base_tag["href"].strip()))
        if urlparse(resolved_base).scheme in ("http", "https"):
            base_url = resolved_base

    seen: set[str] = set()
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        normalized = normalize_url(anchor["href"], base_url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def extract_text(html: str) -> tuple[str, str]:
    """Return ``(title, visible_text)`` for an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return title, " ".join(text.split())


def same_domain(url: str, other: str) -> bool:
    """True when both URLs share a registrable-ish host (``www.`` is ignored)."""
    def host(value: str) -> str:
        netloc = urlparse(value).netloc.lower()
        netloc = netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0]
        return netloc[4:] if netloc.startswith("www.") else netloc

    return host(url) == host(other)


def parse_page(html: str, url: str) -> Page:
    """Parse fetched HTML into the record the crawler stores."""
    title, text = extract_text(html)
    return Page(url=url, title=title, text=text, links=extract_links(html, url))


# --------------------------------------------------------------------------------------
# The crawler
# --------------------------------------------------------------------------------------


class Crawler:
    """Breadth-first, multi-threaded crawler that persists pages to SQLite."""

    def __init__(self, config: Config, conn: sqlite3.Connection | None = None) -> None:
        self.config = config
        self._owns_conn = conn is None
        self.conn = conn if conn is not None else storage.init_db(config.db_path)

        self._state_lock = threading.Lock()  # guards _visited and _budget
        self._db_lock = threading.Lock()  # serialises writes to SQLite
        self._visited: set[str] = set()
        self._budget = config.max_pages
        self.stats = CrawlStats()

        self._seed_urls = [u for u in (normalize_url(u) for u in config.seed_urls) if u]
        self._allowed = {d.lower() for d in config.allowed_domains}
        self._local = threading.local()

    # -- session handling ---------------------------------------------------------

    def _session(self) -> requests.Session:
        """One :class:`requests.Session` per worker thread (Sessions are not thread-safe)."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {"User-Agent": self.config.user_agent, "Accept": "text/html,*/*;q=0.8"}
            )
            self._local.session = session
        return session

    # -- shared state -------------------------------------------------------------

    def _claim(self, url: str) -> bool:
        """Atomically reserve one page of budget for ``url``.

        Returns ``False`` if the URL was already claimed or the budget is exhausted.
        """
        with self._state_lock:
            if self._budget <= 0 or url in self._visited:
                return False
            self._visited.add(url)
            self._budget -= 1
            self.stats.urls_seen += 1
            return True

    def _release(self) -> None:
        """Give a page of budget back after a failed fetch."""
        with self._state_lock:
            self._budget += 1

    def _budget_left(self) -> int:
        with self._state_lock:
            return self._budget

    def is_allowed(self, url: str) -> bool:
        """Domain policy: explicit allow-list wins, otherwise optionally stay on the seeds."""
        if self._allowed:
            host = urlparse(url).netloc.lower()
            host = host.rsplit("@", 1)[-1].rsplit(":", 1)[0]
            bare = host[4:] if host.startswith("www.") else host
            return any(bare == d or bare.endswith(f".{d}") for d in self._allowed)
        if not self.config.same_domain_only:
            return True
        return any(same_domain(url, seed) for seed in self._seed_urls)

    # -- work units ---------------------------------------------------------------

    def fetch(self, url: str) -> Page | None:
        """Fetch and parse a single URL. Returns ``None`` on any non-HTML or error case."""
        try:
            response = self._session().get(
                url, timeout=self.config.request_timeout, allow_redirects=True
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("fetch failed for %s: %s", url, exc)
            return None

        content_type = response.headers.get("Content-Type", "")
        if content_type and "html" not in content_type.lower():
            logger.debug("skipping %s (content-type %s)", url, content_type)
            return None

        # requests falls back to latin-1 when the HTTP header omits a charset, which
        # mangles UTF-8 pages. Sniff the real encoding from the body in that case.
        if "charset=" not in content_type.lower():
            response.encoding = response.apparent_encoding or "utf-8"

        html = response.text
        if len(html) > self.config.max_page_bytes:
            html = html[: self.config.max_page_bytes]

        final_url = normalize_url(response.url) or url
        return parse_page(html, final_url)

    def save(self, page: Page) -> None:
        """Persist a page. Serialised behind a lock so threads never write concurrently."""
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._db_lock, self.conn:
            self.conn.execute(
                "INSERT INTO documents (url, title, text, fetched_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET "
                "title = excluded.title, text = excluded.text, fetched_at = excluded.fetched_at",
                (page.url, page.title, page.text, fetched_at),
            )

    def _work(self, url: str) -> Page | None:
        page = self.fetch(url)
        if page is None:
            return None
        self.save(page)
        return page

    # -- driver -------------------------------------------------------------------

    def crawl(self) -> CrawlStats:
        """Run the crawl until ``max_pages`` documents are stored or the frontier empties."""
        frontier: list[str] = [u for u in self._seed_urls if self.is_allowed(u)]
        if not frontier:
            logger.warning("no crawlable seed URLs in config")
            return self.stats

        # Keep roughly two tasks queued per worker so the pool never starves.
        max_in_flight = max(self.config.max_workers * 2, self.config.max_workers + 1)
        pending: set[Future[Page | None]] = set()

        with ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="crawler"
        ) as executor:
            while frontier or pending:
                while frontier and len(pending) < max_in_flight and self._budget_left() > 0:
                    url = frontier.pop(0)
                    if self._claim(url):
                        pending.add(executor.submit(self._work, url))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        page = future.result()
                    except Exception:  # a worker bug must not kill the crawl
                        logger.exception("crawler worker raised")
                        self.stats.pages_failed += 1
                        self._release()
                        continue

                    if page is None:
                        self.stats.pages_failed += 1
                        self._release()
                        continue

                    self.stats.pages_crawled += 1
                    logger.info(
                        "[%d/%d] %s", self.stats.pages_crawled, self.config.max_pages, page.url
                    )
                    if self._budget_left() > 0:
                        frontier.extend(
                            link
                            for link in page.links
                            if link not in self._visited and self.is_allowed(link)
                        )

                if self._budget_left() <= 0 and not pending:
                    break

        return self.stats

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()


def run_crawl(config: Config) -> CrawlStats:
    """Convenience entry point used by the CLI."""
    config.ensure_db_parent()
    crawler = Crawler(config)
    try:
        return crawler.crawl()
    finally:
        crawler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from config import load_config

    print(run_crawl(load_config()).as_dict())
