"""
Persistent SQLite-backed API response cache for all MCP server calls.

Usage (decorator pattern):
    from pipelines.api_cache import api_cached

    @api_cached(ttl_days=30)
    def query_gtex_eqtl(gene_symbol: str, tissue: str, ...) -> dict:
        ...

Usage (inline):
    from pipelines.api_cache import get_cache
    cache = get_cache()
    result = cache.get_or_set("gtex_eqtl", (gene, tissue), fetch_fn, ttl_days=30)

Cache key = SHA-256 of (fn_name, json-serialised positional + keyword args).
Thread-safe via WAL mode + per-call connection.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "api_cache.sqlite"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key         TEXT    PRIMARY KEY,
    value       TEXT    NOT NULL,
    cached_at   REAL    NOT NULL,
    ttl_seconds INTEGER NOT NULL
);
"""


def _make_key(fn_name: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps([fn_name, list(args), sorted(kwargs.items())],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class ApiCache:
    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def get(self, key: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, cached_at, ttl_seconds FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value_json, cached_at, ttl_seconds = row
        if time.time() > cached_at + ttl_seconds:
            self._last_miss_was_expired = True
            return None
        self._last_miss_was_expired = False
        return json.loads(value_json)

    _last_miss_was_expired: bool = False

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache_entries (key, value, cached_at, ttl_seconds)
                   VALUES (?, ?, ?, ?)""",
                (key, json.dumps(value, default=str), time.time(), ttl_seconds),
            )

    def get_or_set(
        self,
        fn_name: str,
        args: tuple,
        kwargs: dict,
        fetch_fn: Callable,
        ttl_days: int = 30,
    ) -> Any:
        key = _make_key(fn_name, args, kwargs)
        cached = self.get(key)
        if cached is not None:
            return cached
        if self._last_miss_was_expired:
            log.warning("api_cache: expired entry hit — re-fetching live  fn=%s  args=%s", fn_name, args)
        result = fetch_fn()
        if result is not None:
            self.set(key, result, ttl_days * 86400)
        return result

    def invalidate(self, fn_name: str, args: tuple, kwargs: dict) -> None:
        key = _make_key(fn_name, args, kwargs)
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE cached_at + ttl_seconds < ?",
                (time.time(),),
            ).fetchone()[0]
        return {"total_entries": total, "expired_entries": expired, "live_entries": total - expired}


_cache_singleton: ApiCache | None = None


def get_cache(db_path: Path | None = None) -> ApiCache:
    global _cache_singleton
    if _cache_singleton is None or db_path is not None:
        _cache_singleton = ApiCache(db_path or _DEFAULT_DB)
    return _cache_singleton


def api_cached(ttl_days: int = 30):
    """
    Decorator: caches the return value of an API function by its arguments.
    The function must return a JSON-serialisable dict or None.
    A None return (error) is not cached so the next call retries the live API.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            return cache.get_or_set(
                fn.__name__,
                args,
                kwargs,
                lambda: fn(*args, **kwargs),
                ttl_days=ttl_days,
            )
        return wrapper
    return decorator
