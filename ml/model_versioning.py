"""Helpers for managing persisted ML model metadata and compatibility checks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from packaging.version import InvalidVersion, Version

try:  # pragma: no cover - optional runtime dependency in some environments
    import sklearn
except Exception:  # pragma: no cover - defensive, we only care if import works
    sklearn = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

_METADATA_SUFFIX = "_metadata.json"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompatibilityCheck:
    """Result of a compatibility validation."""

    ok: bool
    reason: Optional[str] = None


def _metadata_path(model_dir: Path, model_name: str) -> Path:
    return model_dir / f"{model_name}{_METADATA_SUFFIX}"


def load_metadata(model_dir: Path | str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load metadata JSON for *model_name* if it exists."""

    path = _metadata_path(Path(model_dir), model_name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("failed to read model metadata %s: %s", path, exc)
        return None
    if not isinstance(data, Mapping):
        LOGGER.warning("model metadata %s is not a mapping", path)
        return None
    return dict(data)


def save_metadata(
    model_dir: Path | str,
    model_name: str,
    *,
    extra: Optional[Mapping[str, Any]] = None,
    artifact_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Persist metadata for a model and return the stored payload."""

    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = _metadata_path(directory, model_name)
    previous = load_metadata(directory, model_name) or {}
    version = int(previous.get("model_version", 0) or 0) + 1

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_version": version,
        "schema_version": _SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    if sklearn is not None:
        payload["scikit_learn_version"] = getattr(sklearn, "__version__", "unknown")
    if artifact_path is not None:
        payload["artifact"] = str(Path(artifact_path))
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})

    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        LOGGER.error("failed to write model metadata %s: %s", path, exc)
        raise
    return payload


def check_sklearn_compatibility(metadata: Optional[Mapping[str, Any]]) -> CompatibilityCheck:
    """Validate metadata against the currently installed scikit-learn version."""

    if metadata is None:
        return CompatibilityCheck(ok=True)
    saved_version = metadata.get("scikit_learn_version")
    if sklearn is None or not saved_version:
        return CompatibilityCheck(ok=True)
    try:
        saved = Version(str(saved_version))
        current = Version(str(getattr(sklearn, "__version__", "0")))
    except InvalidVersion as exc:
        return CompatibilityCheck(ok=False, reason=f"invalid version metadata: {exc}")

    if (saved.major, saved.minor) != (current.major, current.minor):
        reason = (
            "scikit-learn version mismatch: "
            f"saved={saved.major}.{saved.minor}, current={current.major}.{current.minor}"
        )
        return CompatibilityCheck(ok=False, reason=reason)
    return CompatibilityCheck(ok=True)


def record_successful_load(model_dir: Path | str, model_name: str) -> None:
    """Update metadata to reflect a successful load."""

    path = _metadata_path(Path(model_dir), model_name)
    metadata = load_metadata(model_dir, model_name)
    if metadata is None:
        return
    metadata["last_loaded_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except OSError as exc:  # pragma: no cover - logging only
        LOGGER.warning("failed to update model metadata %s: %s", path, exc)


__all__ = [
    "CompatibilityCheck",
    "check_sklearn_compatibility",
    "load_metadata",
    "record_successful_load",
    "save_metadata",
]
