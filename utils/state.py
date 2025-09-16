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

        self.metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"completed", "failed"}
        }

    def _flush(self) -> None:
        io.ensure_dir(self.path.parent)
        payload = {
            **self.metadata,
            "completed": self.completed,
            "failed": self.failed,
            "updated_at": time.time(),
        }
        io.write_json(self.path, payload)

    # ------------------------------------------------------------------
    def set_metadata(self, **fields: Any) -> None:
        if not fields:
            return
        self.metadata.update(fields)
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
        if extra:
            bucket = self.metadata.setdefault("completed_extra", {})
            if isinstance(bucket, dict):
                bucket[key] = extra
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
        self._flush()

    def stats(self) -> Dict[str, Any]:
        return {
            "completed": len(self.completed),
            "failed": len(self.failed),
            "state_path": str(self.path),
        }
