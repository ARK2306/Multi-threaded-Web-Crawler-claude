"""Configuration loading for the mini search engine.

The config lives outside the repository (by default ``~/.secrets/search_engine_config.json``)
so that seed lists and database locations are never committed. Individual values can be
overridden with environment variables, which is how the Docker image points the database
at the ``/data`` volume without editing the JSON file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("~/.secrets/search_engine_config.json")

# Environment overrides. Every one of these is optional.
ENV_CONFIG_PATH = "SEARCH_ENGINE_CONFIG"
ENV_DB_PATH = "SEARCH_ENGINE_DB_PATH"
ENV_MAX_PAGES = "SEARCH_ENGINE_MAX_PAGES"
ENV_SEED_URLS = "SEARCH_ENGINE_SEED_URLS"  # comma separated
ENV_MAX_WORKERS = "SEARCH_ENGINE_MAX_WORKERS"


class ConfigError(RuntimeError):
    """Raised when the config file is missing, malformed, or incomplete."""


@dataclass
class Config:
    """Runtime configuration.

    ``seed_urls``, ``max_pages`` and ``db_path`` are required; the rest have defaults.
    """

    seed_urls: list[str]
    max_pages: int
    db_path: Path
    max_workers: int = 8
    request_timeout: float = 10.0
    same_domain_only: bool = True
    max_page_bytes: int = 2_000_000
    user_agent: str = "MiniSearchBot/1.0"
    allowed_domains: list[str] = field(default_factory=list)

    def ensure_db_parent(self) -> None:
        """Create the directory that will hold the SQLite file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def config_path() -> Path:
    raw = os.environ.get(ENV_CONFIG_PATH) or str(DEFAULT_CONFIG_PATH)
    return Path(raw).expanduser()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(
            f"Config file not found at {path}. Create it with keys "
            '"seed_urls", "max_pages" and "db_path", or set '
            f"{ENV_CONFIG_PATH} to point somewhere else."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object.")
    return data


def _split_urls(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_config(path: Path | str | None = None) -> Config:
    """Load the config file, apply environment overrides, and validate the result."""
    resolved = Path(path).expanduser() if path is not None else config_path()
    data = _read_json(resolved)

    seed_urls = data.get("seed_urls")
    if env_seeds := os.environ.get(ENV_SEED_URLS):
        seed_urls = _split_urls(env_seeds)
    if not isinstance(seed_urls, list) or not seed_urls:
        raise ConfigError(f'"seed_urls" in {resolved} must be a non-empty list of URLs.')
    seed_urls = [str(url).strip() for url in seed_urls]

    max_pages = os.environ.get(ENV_MAX_PAGES, data.get("max_pages"))
    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f'"max_pages" in {resolved} must be an integer.') from exc
    if max_pages < 1:
        raise ConfigError('"max_pages" must be at least 1.')

    db_path = os.environ.get(ENV_DB_PATH, data.get("db_path"))
    if not db_path:
        raise ConfigError(f'"db_path" is missing from {resolved}.')

    max_workers = os.environ.get(ENV_MAX_WORKERS, data.get("max_workers", 8))
    try:
        max_workers = max(1, int(max_workers))
    except (TypeError, ValueError) as exc:
        raise ConfigError('"max_workers" must be an integer.') from exc

    return Config(
        seed_urls=seed_urls,
        max_pages=max_pages,
        db_path=Path(str(db_path)).expanduser(),
        max_workers=max_workers,
        request_timeout=float(data.get("request_timeout", 10.0)),
        same_domain_only=bool(data.get("same_domain_only", True)),
        max_page_bytes=int(data.get("max_page_bytes", 2_000_000)),
        user_agent=str(data.get("user_agent", "MiniSearchBot/1.0")),
        allowed_domains=[str(d) for d in data.get("allowed_domains", [])],
    )
