from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


class SQLiteCache:
    """Simple TTL cache backed by SQLite."""

    def __init__(self, db_path: str, ttl_days: int) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._ttl_seconds = max(0.0, float(ttl_days)) * 86400.0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._configure_connection()
        self._last_purge = 0.0

    def _configure_connection(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON cache (timestamp)"
                )
            except sqlite3.DatabaseError as exc:
                LOGGER.warning("Failed to initialize cache database %s: %s", self._db_path, exc)

    def get(self, key: str) -> Optional[Any]:
        """Return cached value for key if not expired."""
        if not key:
            return None

        self._maybe_purge()

        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value, timestamp FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            LOGGER.warning("Failed to read cache for %s: %s", key, exc)
            return None

        if not row:
            return None

        value_raw, timestamp = row
        if self._ttl_seconds > 0 and time.time() - float(timestamp) > self._ttl_seconds:
            self._delete(key)
            return None

        try:
            return json.loads(value_raw)
        except json.JSONDecodeError:
            LOGGER.warning("Invalid JSON payload in cache for key %s", key)
            self._delete(key)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache under key."""
        if not key:
            return

        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            LOGGER.debug("Value for key %s is not JSON serializable: %s", key, exc)
            return

        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, serialized, now),
                )
        except sqlite3.DatabaseError as exc:
            LOGGER.warning("Failed to write cache for %s: %s", key, exc)
            return

        self._maybe_purge()

    def _delete(self, key: str) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            except sqlite3.DatabaseError:
                LOGGER.debug("Failed to delete cache key %s", key)

    def _maybe_purge(self) -> None:
        if self._ttl_seconds <= 0:
            return

        now = time.time()
        interval = max(60.0, min(3600.0, self._ttl_seconds / 4 or 60.0))
        if now - self._last_purge < interval:
            return

        self._purge_expired()

    def _purge_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return

        cutoff = time.time() - self._ttl_seconds
        try:
            with self._lock:
                self._conn.execute("DELETE FROM cache WHERE timestamp <= ?", (cutoff,))
        except sqlite3.DatabaseError as exc:
            LOGGER.debug("Failed to purge expired cache entries: %s", exc)
        else:
            self._last_purge = time.time()
