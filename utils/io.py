"""I/O helpers with pathlib-friendly APIs and safe writes."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

LOGGER = logging.getLogger("utils.io")


class IoError(RuntimeError):
    """Raised when a filesystem operation fails."""


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("failed to create directory %s: %s", directory, exc)
        raise IoError(f"unable to create directory: {directory}") from exc
    return directory


def atomic_write(path: os.PathLike[str] | str, data: bytes) -> Path:
    """Atomically write bytes to *path* using a temporary file."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, dir=target.parent) as handle:
            handle.write(data)
            tmp_path = Path(handle.name)
        tmp_path.replace(target)
        return target
    except Exception as exc:
        LOGGER.error("atomic write failed for %s: %s", target, exc)
        raise IoError(f"atomic write failed for {target}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.debug("temporary file cleanup failed for %s", tmp_path)


def atomic_write_iter(path: os.PathLike[str] | str, chunks: Iterable[bytes]) -> Path:
    """Atomically write a stream of *chunks* to *path*."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, dir=target.parent) as handle:
            for chunk in chunks:
                handle.write(chunk)
            tmp_path = Path(handle.name)
        tmp_path.replace(target)
        return target
    except Exception as exc:
        LOGGER.error("atomic stream write failed for %s: %s", target, exc)
        raise IoError(f"atomic stream write failed for {target}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.debug("temporary file cleanup failed for %s", tmp_path)




def read_text(path: os.PathLike[str] | str, *, encoding: str = "utf-8") -> str:
    """Read text content from *path*."""
    target = Path(path)
    try:
        return target.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError) as exc:
        LOGGER.error("failed to read text from %s: %s", target, exc)
        raise IoError(f"unable to read text from {target}") from exc


def read_json(path: os.PathLike[str] | str, *, default: object | None = None, encoding: str = "utf-8") -> object:
    """Load JSON from *path* if it exists, otherwise return *default*."""
    target = Path(path)
    if not target.exists():
        return default
    try:
        with target.open("r", encoding=encoding) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("failed to read json from %s: %s", target, exc)
        return default

def write_text(path: os.PathLike[str] | str, content: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write(path, content.encode(encoding))


def write_json(path: os.PathLike[str] | str, obj: object) -> Path:
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    return write_text(path, data)


def log_json(path: os.PathLike[str] | str, obj: Mapping[str, object]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    line = json.dumps(obj, ensure_ascii=False)
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        LOGGER.error("failed to append log entry to %s: %s", target, exc)
        raise IoError(f"unable to append log entry to {target}") from exc


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_writer(
    path: os.PathLike[str] | str,
    rows: Iterable[Iterable[object]],
    *,
    header: Iterable[object] | None = None,
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header is not None:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return target


def now_iso() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat()
