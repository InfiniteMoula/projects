
"""HTTP convenience helpers."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from functools import wraps
from typing import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
)

import httpx

from utils import io

LOGGER = logging.getLogger("utils.http")
T = TypeVar("T")


class HttpError(RuntimeError):
    """Raised when an HTTP operation fails."""


def retry_on_network_error(
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    retry_statuses: Optional[Sequence[int]] = None,
    retry_exceptions: Optional[Sequence[type[BaseException]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a network call on transient errors with exponential backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be >= 0")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be >= 1")

    default_statuses = (403, 408, 409, 425, 429, 500, 502, 503, 504)
    retry_status_set = set(int(code) for code in (retry_statuses or default_statuses))

    default_exceptions: list[type[BaseException]] = [HttpError, httpx.RequestError, TimeoutError]
    try:
        import requests as _requests  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _requests = None  # type: ignore
    else:
        default_exceptions.append(_requests.RequestException)  # type: ignore[attr-defined]

    if retry_exceptions:
        exception_types = tuple(set(list(default_exceptions) + list(retry_exceptions)))
    else:
        exception_types = tuple(default_exceptions)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            attempt = 1
            delay = initial_delay
            while True:
                try:
                    result = func(*args, **kwargs)
                except exception_types as exc:
                    if attempt >= max_attempts:
                        raise
                    LOGGER.warning(
                        "Retrying %s after %s (attempt %d/%d) in %.2fs",
                        func.__name__,
                        exc,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    if delay:
                        time.sleep(delay)
                    attempt += 1
                    delay *= backoff_factor
                    continue

                status_code = getattr(result, "status_code", None)
                if (
                    isinstance(status_code, int)
                    and status_code in retry_status_set
                ):
                    if attempt >= max_attempts:
                        close_candidate = getattr(result, "close", None)
                        if callable(close_candidate):
                            try:
                                close_candidate()
                            except Exception:  # pragma: no cover - best effort cleanup
                                pass
                        raise HttpError(
                            f"{func.__name__} returned HTTP {status_code} after {attempt} attempts"
                        )
                    LOGGER.warning(
                        "Retrying %s due to HTTP %s (attempt %d/%d) in %.2fs",
                        func.__name__,
                        status_code,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    close_candidate = getattr(result, "close", None)
                    if callable(close_candidate):
                        try:
                            close_candidate()
                        except Exception:  # pragma: no cover - best effort cleanup
                            pass
                    if delay:
                        time.sleep(delay)
                    attempt += 1
                    delay *= backoff_factor
                    continue

                return result

        return wrapper

    return decorator


def request_with_backoff(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_attempts: int = 5,
    backoff_factor: float = 0.5,
    backoff_max: float = 8.0,
    retry_statuses: Optional[Iterable[int]] = None,
    request_tracker: Optional[Callable[[int], None]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: object,
) -> httpx.Response:
    """Perform an HTTP request with exponential backoff and jitter."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    retry_codes = set(retry_statuses or (408, 409, 425, 429, 500, 502, 503, 504))
    log = logger or LOGGER
    last_exc: Optional[httpx.HTTPError] = None
    last_response: Optional[httpx.Response] = None
    request_kwargs = dict(kwargs) if kwargs else {}

    request_method = getattr(client, "request", None)
    if callable(request_method):
        def _perform_request() -> httpx.Response:
            return request_method(method, url, **request_kwargs)
    else:
        fallback = getattr(client, method.lower(), None)
        if not callable(fallback):
            raise HttpError(f"client {client!r} does not support method {method}")

        def _perform_request() -> httpx.Response:
            return fallback(url, **request_kwargs)

    for attempt in range(1, max_attempts + 1):
        try:
            response = _perform_request()
        except httpx.HTTPError as exc:
            last_exc = exc
            if request_tracker:
                request_tracker(0)
            if attempt >= max_attempts:
                raise HttpError(f"{method} {url} failed after {attempt} attempts: {exc}") from exc
            base_delay = min(backoff_max, backoff_factor * (2 ** (attempt - 1)))
            sleep_for = base_delay * random.uniform(0.5, 1.0) if base_delay else 0.0
            log.warning(
                "HTTP %s %s failed (attempt %d/%d): %s; retrying in %.2fs",
                method,
                url,
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            if sleep_for > 0:
                time.sleep(sleep_for)
            continue

        last_response = response
        if request_tracker:
            try:
                response_bytes = response.content or b""
            except Exception:
                response_bytes = b""
            request_tracker(len(response_bytes))
        if retry_codes and response.status_code in retry_codes and attempt < max_attempts:
            base_delay = min(backoff_max, backoff_factor * (2 ** (attempt - 1)))
            sleep_for = base_delay * random.uniform(0.5, 1.0) if base_delay else 0.0
            log.warning(
                "HTTP %s %s returned %s (attempt %d/%d); retrying in %.2fs",
                method,
                url,
                response.status_code,
                attempt,
                max_attempts,
                sleep_for,
            )
            if sleep_for > 0:
                time.sleep(sleep_for)
            continue

        return response

    if last_response is not None:
        return last_response

    assert last_exc is not None
    raise HttpError(f"{method} {url} failed") from last_exc


def stream_download(
    url: str,
    dest: Path | str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    resume: bool = True,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> Path:
    """Download *url* to *dest* using atomic writes and optional resume."""

    target = Path(dest)
    io.ensure_dir(target.parent)
    request_headers: MutableMapping[str, str] = dict(headers or {})
    existing_size = target.stat().st_size if target.exists() else 0
    if resume and existing_size > 0:
        request_headers.setdefault("Range", f"bytes={existing_size}-")

    def chunk_iter() -> Iterable[bytes]:
        total_bytes = 0
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
                        total_bytes += len(chunk)
                        yield chunk
                
                # Track the request after completion
                if request_tracker:
                    request_tracker(total_bytes)
                    
        except httpx.HTTPError as exc:
            # Still track the request even on failure  
            if request_tracker:
                request_tracker(total_bytes)
            raise HttpError(f"download failed for {url}: {exc}") from exc

    return io.atomic_write_iter(target, chunk_iter())


@retry_on_network_error()
def get_json(
    url: str,
    *,
    params: Mapping[str, object] | None = None,
    timeout: float = 20.0,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> object:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            
            # Track the request
            response_size = len(response.content) if response.content else 0
            if request_tracker:
                request_tracker(response_size)
                
            return response.json()
    except httpx.HTTPError as exc:
        # Still track the request even on failure
        if request_tracker:
            request_tracker(0)
        raise HttpError(f"GET {url} failed: {exc}") from exc
