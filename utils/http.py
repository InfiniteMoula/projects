
"""HTTP convenience helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import httpx

from utils import io

LOGGER = logging.getLogger("utils.http")


class HttpError(RuntimeError):
    """Raised when an HTTP operation fails."""


def stream_download(
    url: str,
    dest: Path | str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    resume: bool = True,
) -> Path:
    """Download *url* to *dest* using atomic writes and optional resume."""

    target = Path(dest)
    io.ensure_dir(target.parent)
    request_headers: MutableMapping[str, str] = dict(headers or {})
    existing_size = target.stat().st_size if target.exists() else 0
    if resume and existing_size > 0:
        request_headers.setdefault("Range", f"bytes={existing_size}-")

    def chunk_iter() -> Iterable[bytes]:
        try:
            with httpx.stream(
                "GET",
                url,
                headers=request_headers,
                follow_redirects=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                if existing_size and response.status_code == 206:
                    with target.open("rb") as existing:
                        for chunk in iter(lambda: existing.read(1 << 20), b""):
                            yield chunk
                elif existing_size and response.status_code != 206:
                    LOGGER.info("server ignored range request for %s, restarting download", url)
                for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise HttpError(f"download failed for {url}: {exc}") from exc

    return io.atomic_write_iter(target, chunk_iter())


def get_json(
    url: str,
    *,
    params: Mapping[str, object] | None = None,
    timeout: float = 20.0,
) -> object:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HttpError(f"GET {url} failed: {exc}") from exc
