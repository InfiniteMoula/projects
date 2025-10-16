from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedResponse:
    """Cached HTTP response payload stored in SQLite."""

    status: int
    headers: Dict[str, str]
    payload: bytes


class SqliteCache:
    """Persistent HTTP cache with TTL and LRU eviction."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS http_cache (
            key TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            method TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            status INTEGER NOT NULL,
            headers TEXT NOT NULL,
            body BLOB NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            last_accessed REAL NOT NULL
        )
    """

    _CREATE_INDEX = (
        "CREATE INDEX IF NOT EXISTS idx_http_cache_key_expires ON http_cache (key, expires_at)"
    )

    def __init__(self, db_path: str, ttl_seconds: int, max_items: int) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._ttl_seconds = max(0, int(ttl_seconds))
        self._max_items = max(0, int(max_items))
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._configure()

        self._cleanup_interval = max(60.0, min(3600.0, self._ttl_seconds / 4 or 60.0))
        self._last_cleanup = 0.0

    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        except sqlite3.DatabaseError as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to open cache database %s: %s", self._db_path, exc)
            raise

    def _configure(self) -> None:
        try:
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute(self._CREATE_TABLE)
                self._conn.execute(self._CREATE_INDEX)
        except sqlite3.DatabaseError as exc:
            LOGGER.warning("Failed to configure cache database %s: %s", self._db_path, exc)
            self._handle_corruption(exc)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.DatabaseError:
                LOGGER.debug("Error closing SQLite cache connection", exc_info=True)

    def get(self, url: str, method: str, params_hash: str) -> Optional[CachedResponse]:
        key = self._make_key(url, method, params_hash)
        now = time.time()
        self._maybe_cleanup(now)
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT status, headers, body FROM http_cache WHERE key = ? AND expires_at > ?",
                    (key, now),
                ).fetchone()
                if not row:
                    return None
                status, headers_raw, body = row
                self._conn.execute(
                    "UPDATE http_cache SET last_accessed = ? WHERE key = ?",
                    (now, key),
                )
        except sqlite3.DatabaseError as exc:
            LOGGER.warning("SQLite cache read failed for %s: %s", key, exc)
            self._handle_corruption(exc)
            return None

        try:
            headers_obj = json.loads(headers_raw)
        except json.JSONDecodeError:
            LOGGER.debug("Invalid headers JSON for cached key %s", key)
            self.delete(key)
            return None

        if not isinstance(headers_obj, dict):
            LOGGER.debug("Unexpected headers type for cached key %s", key)
            self.delete(key)
            return None

        headers = {str(k): str(v) for k, v in headers_obj.items()}
        payload = bytes(body)
        return CachedResponse(status=int(status), headers=headers, payload=payload)

    def set(
        self,
        url: str,
        method: str,
        params_hash: str,
        payload: bytes,
        headers: Dict[str, str],
        status: int,
    ) -> None:
        if self._ttl_seconds == 0:
            return

        key = self._make_key(url, method, params_hash)
        now = time.time()
        expires_at = now + float(self._ttl_seconds)
        headers_json = json.dumps({str(k): str(v) for k, v in dict(headers).items()}, ensure_ascii=False)

        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO http_cache (
                        key, url, method, params_hash, status, headers, body, created_at, expires_at, last_accessed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        url = excluded.url,
                        method = excluded.method,
                        params_hash = excluded.params_hash,
                        status = excluded.status,
                        headers = excluded.headers,
                        body = excluded.body,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        last_accessed = excluded.last_accessed
                    """,
                    (
                        key,
                        url,
                        method.upper(),
                        params_hash,
                        int(status),
                        headers_json,
                        sqlite3.Binary(payload),
                        now,
                        expires_at,
                        now,
                    ),
                )
                self._purge_expired_locked(now)
                self._enforce_limit_locked()
        except sqlite3.DatabaseError as exc:
            LOGGER.warning("SQLite cache write failed for %s: %s", key, exc)
            self._handle_corruption(exc)

    def delete(self, key: str) -> None:
        try:
            with self._lock:
                self._conn.execute("DELETE FROM http_cache WHERE key = ?", (key,))
        except sqlite3.DatabaseError:
            LOGGER.debug("Failed to delete cache key %s", key, exc_info=True)

    def _purge_expired_locked(self, now: float) -> None:
        if self._ttl_seconds <= 0:
            return
        try:
            self._conn.execute("DELETE FROM http_cache WHERE expires_at <= ?", (now,))
        except sqlite3.DatabaseError as exc:
            LOGGER.debug("Failed to purge expired cache rows: %s", exc)

    def _enforce_limit_locked(self) -> None:
        if self._max_items <= 0:
            return
        try:
            count_row = self._conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()
            if not count_row:
                return
            count = int(count_row[0])
            if count <= self._max_items:
                return
            overflow = count - self._max_items
            self._conn.execute(
                "DELETE FROM http_cache WHERE key IN ("
                "SELECT key FROM http_cache ORDER BY last_accessed ASC LIMIT ?"
                ")",
                (overflow,),
            )
        except sqlite3.DatabaseError as exc:
            LOGGER.debug("Failed to enforce cache size: %s", exc)

    def _maybe_cleanup(self, now: float) -> None:
        if self._ttl_seconds <= 0:
            return
        if now - self._last_cleanup < self._cleanup_interval:
            return
        with self._lock:
            self._purge_expired_locked(now)
        self._last_cleanup = now

    def _handle_corruption(self, exc: Exception) -> None:
        LOGGER.error("Resetting SQLite cache due to database error: %s", exc)
        try:
            with self._lock:
                try:
                    self._conn.close()
                except sqlite3.DatabaseError:
                    pass
                backup_path = self._db_path.with_suffix(self._db_path.suffix + ".corrupt")
                try:
                    if self._db_path.exists():
                        if backup_path.exists():
                            backup_path.unlink()
                        self._db_path.rename(backup_path)
                except OSError:
                    try:
                        self._db_path.unlink()
                    except OSError:
                        LOGGER.debug("Unable to remove corrupt cache file %s", self._db_path)
                self._conn = self._connect()
                self._configure()
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Failed to reset SQLite cache after corruption")

    @staticmethod
    def _make_key(url: str, method: str, params_hash: str) -> str:
        digest = hashlib.sha256()
        digest.update(method.upper().encode("utf-8"))
        digest.update(b"\0")
        digest.update(url.encode("utf-8"))
        digest.update(b"\0")
        digest.update(params_hash.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def hash_payload(payload: Optional[bytes]) -> str:
        """Return a deterministic hash for request payloads used in cache keys."""
        data = payload or b""
        return hashlib.sha256(data).hexdigest()
