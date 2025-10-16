from __future__ import annotations

import time
from pathlib import Path

from cache.sqlite_cache import CachedResponse, SqliteCache


def _hash(data: bytes) -> str:
    return SqliteCache.hash_payload(data)


def test_sqlite_cache_hit_and_miss(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    cache = SqliteCache(str(db_path), ttl_seconds=60, max_items=10)
    url = "https://example.com/resource"
    method = "GET"
    params_hash = _hash(b"")

    assert cache.get(url, method, params_hash) is None

    payload = b"payload"
    headers = {"Content-Type": "text/plain"}
    cache.set(url, method, params_hash, payload, headers, 200)

    cached = cache.get(url, method, params_hash)
    assert isinstance(cached, CachedResponse)
    assert cached.status == 200
    assert cached.headers["Content-Type"] == "text/plain"
    assert cached.payload == payload

    cache.close()


def test_sqlite_cache_ttl_expiration(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    cache = SqliteCache(str(db_path), ttl_seconds=1, max_items=10)
    url = "https://example.com/page"
    method = "GET"
    params_hash = _hash(b"")

    cache.set(url, method, params_hash, b"ttl", {}, 200)
    assert cache.get(url, method, params_hash) is not None

    time.sleep(1.1)
    assert cache.get(url, method, params_hash) is None

    cache.close()


def test_sqlite_cache_lru_eviction(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    cache = SqliteCache(str(db_path), ttl_seconds=60, max_items=2)

    entries = []
    for index in range(2):
        url = f"https://example.com/{index}"
        params_hash = _hash(f"body-{index}".encode("utf-8"))
        payload = f"payload-{index}".encode("utf-8")
        cache.set(url, "GET", params_hash, payload, {}, 200)
        entries.append((url, params_hash))

    # Access first entry to ensure the second becomes the least recently used
    assert cache.get(entries[0][0], "GET", entries[0][1]) is not None

    # Insert a third entry which should evict the second one (LRU)
    url3 = "https://example.com/3"
    params_hash3 = _hash(b"body-3")
    cache.set(url3, "GET", params_hash3, b"payload-3", {}, 200)

    assert cache.get(entries[0][0], "GET", entries[0][1]) is not None
    assert cache.get(entries[1][0], "GET", entries[1][1]) is None
    assert cache.get(url3, "GET", params_hash3) is not None

    cache.close()


def test_sqlite_cache_handles_corruption(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    cache = SqliteCache(str(db_path), ttl_seconds=60, max_items=10)
    url = "https://example.com/corrupt"
    params_hash = _hash(b"")
    cache.set(url, "GET", params_hash, b"payload", {}, 200)
    cache.close()

    db_path.write_bytes(b"corrupt")

    cache = SqliteCache(str(db_path), ttl_seconds=60, max_items=10)
    assert cache.get(url, "GET", params_hash) is None

    cache.set(url, "GET", params_hash, b"fresh", {}, 200)
    cached = cache.get(url, "GET", params_hash)
    assert isinstance(cached, CachedResponse)
    assert cached.payload == b"fresh"

    cache.close()
