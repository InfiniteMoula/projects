from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Any, Dict

from utils import io

LOGGER = logging.getLogger("utils.state")


class SequentialRunState:
    """Persist simple sequential progress information to resume runs."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.completed: list[str] = []
        self.failed: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.in_progress: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        payload = io.read_json(self.path, default=None)
        if not isinstance(payload, dict):
            return

        completed = payload.get("completed")
        if isinstance(completed, list):
            # preserve insertion order while removing duplicates
            seen: set[str] = set()
            self.completed = []
            for item in completed:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    self.completed.append(key)

        failed = payload.get("failed")
        if isinstance(failed, dict):
            cleaned: Dict[str, Dict[str, Any]] = {}
            for key, value in failed.items():
                if isinstance(value, dict):
                    cleaned[str(key)] = dict(value)
            self.failed = cleaned

        in_progress = payload.get("in_progress")
        if isinstance(in_progress, list):
            self.in_progress = [str(item) for item in in_progress if str(item)]

        self.metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"completed", "failed", "in_progress"}
        }
        self._refresh_progress_metadata()

    def _flush(self) -> None:
        io.ensure_dir(self.path.parent)
        self._refresh_progress_metadata()
        payload = {
            **self.metadata,
            "completed": self.completed,
            "failed": self.failed,
            "in_progress": self.in_progress,
            "updated_at": time.time(),
        }
        io.write_json(self.path, payload)

    # ------------------------------------------------------------------
    def set_metadata(self, **fields: Any) -> None:
        if not fields:
            return
        self.metadata.update(fields)
        self._flush()

    def mark_started(self, item: str) -> None:
        """Track that *item* processing has begun."""
        key = str(item)
        if key not in self.in_progress:
            self.in_progress.append(key)
        self.metadata["last_started"] = key
        self._flush()

    def pending(self, items: Iterable[str]) -> list[str]:
        completed = set(self.completed)
        return [str(item) for item in items if str(item) not in completed]

    def mark_completed(self, item: str, *, extra: Dict[str, Any] | None = None) -> None:
        key = str(item)
        if key not in self.completed:
            self.completed.append(key)
        if key in self.failed:
            self.failed.pop(key, None)
        if key in self.in_progress:
            self.in_progress.remove(key)
        if extra:
            bucket = self.metadata.setdefault("completed_extra", {})
            if isinstance(bucket, dict):
                bucket[key] = extra
        self.metadata["last_completed"] = key
        self._flush()

    def mark_failed(self, item: str, error: str, *, attempt: int | None = None) -> None:
        key = str(item)
        record = self.failed.get(key, {}) if isinstance(self.failed.get(key), dict) else {}
        attempts = record.get("attempts")
        if not isinstance(attempts, int):
            attempts = 0
        record.update(
            {
                "error": error,
                "attempts": attempt if attempt is not None else attempts + 1,
                "last_attempt": time.time(),
            }
        )
        self.failed[key] = record
        if key in self.in_progress:
            self.in_progress.remove(key)
        self.metadata["last_error"] = key
        self._flush()

    def stats(self) -> Dict[str, Any]:
        return {
            "completed": len(self.completed),
            "failed": len(self.failed),
            "state_path": str(self.path),
        }

    # ------------------------------------------------------------------
    def _refresh_progress_metadata(self) -> None:
        completed_count = len(self.completed)
        failed_count = len(self.failed)
        in_progress_count = len(self.in_progress)
        self.metadata.setdefault("completed_count", completed_count)
        self.metadata.setdefault("failed_count", failed_count)
        self.metadata.setdefault("in_progress_count", in_progress_count)
        self.metadata["completed_count"] = completed_count
        self.metadata["failed_count"] = failed_count
        self.metadata["in_progress_count"] = in_progress_count
        total = self.metadata.get("total")
        if isinstance(total, int) and total > 0:
            self.metadata["progress_pct"] = round((completed_count / total) * 100, 2)
