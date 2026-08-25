"""Config loading, validation, and environment overrides."""

from __future__ import annotations

import json

import pytest

import config as config_module
from config import ConfigError, load_config

VALID = {
    "seed_urls": ["https://example.com"],
    "max_pages": 10,
    "db_path": "/tmp/search.db",
}


def write_config(tmp_path, data) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data) if isinstance(data, dict) else data, encoding="utf-8")
    return str(path)


class TestLoadConfig:
    def test_reads_required_fields(self, tmp_path):
        cfg = load_config(write_config(tmp_path, VALID))
        assert cfg.seed_urls == ["https://example.com"]
        assert cfg.max_pages == 10
        assert str(cfg.db_path) == "/tmp/search.db"

    def test_applies_defaults(self, tmp_path):
        cfg = load_config(write_config(tmp_path, VALID))
        assert cfg.max_workers == 8
        assert cfg.same_domain_only is True
        assert cfg.request_timeout == 10.0

    def test_optional_fields_are_read(self, tmp_path):
        cfg = load_config(write_config(tmp_path, {**VALID, "max_workers": 3, "same_domain_only": False}))
        assert cfg.max_workers == 3
        assert cfg.same_domain_only is False

    def test_expands_home_in_db_path(self, tmp_path):
        cfg = load_config(write_config(tmp_path, {**VALID, "db_path": "~/db/search.db"}))
        assert "~" not in str(cfg.db_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.json")

    def test_malformed_json(self, tmp_path):
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(write_config(tmp_path, "{not json"))

    @pytest.mark.parametrize("key", ["seed_urls", "max_pages", "db_path"])
    def test_missing_required_key(self, tmp_path, key):
        data = {k: v for k, v in VALID.items() if k != key}
        with pytest.raises(ConfigError):
            load_config(write_config(tmp_path, data))

    def test_empty_seed_list(self, tmp_path):
        with pytest.raises(ConfigError, match="non-empty"):
            load_config(write_config(tmp_path, {**VALID, "seed_urls": []}))

    def test_non_integer_max_pages(self, tmp_path):
        with pytest.raises(ConfigError, match="integer"):
            load_config(write_config(tmp_path, {**VALID, "max_pages": "many"}))

    def test_zero_max_pages(self, tmp_path):
        with pytest.raises(ConfigError, match="at least 1"):
            load_config(write_config(tmp_path, {**VALID, "max_pages": 0}))

    def test_ensure_db_parent_creates_directory(self, tmp_path):
        cfg = load_config(write_config(tmp_path, {**VALID, "db_path": str(tmp_path / "nested/deep/s.db")}))
        cfg.ensure_db_parent()
        assert (tmp_path / "nested/deep").is_dir()


class TestEnvironmentOverrides:
    def test_db_path_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(config_module.ENV_DB_PATH, "/data/other.db")
        assert str(load_config(write_config(tmp_path, VALID)).db_path) == "/data/other.db"

    def test_max_pages_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(config_module.ENV_MAX_PAGES, "42")
        assert load_config(write_config(tmp_path, VALID)).max_pages == 42

    def test_seed_urls_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(config_module.ENV_SEED_URLS, "https://a.com, https://b.com")
        assert load_config(write_config(tmp_path, VALID)).seed_urls == ["https://a.com", "https://b.com"]

    def test_max_workers_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(config_module.ENV_MAX_WORKERS, "16")
        assert load_config(write_config(tmp_path, VALID)).max_workers == 16

    def test_config_path_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(config_module.ENV_CONFIG_PATH, write_config(tmp_path, VALID))
        assert load_config().max_pages == 10
