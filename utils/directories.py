"""Directory initialization helpers for pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from . import io

DEFAULT_PIPELINE_DIRECTORIES: tuple[str, ...] = (
    "normalized",
    "contacts",
    "logs",
    "reports",
    "metrics",
)


def initialize_pipeline_directories(
    base_dir: Path | str,
    extra_dirs: Sequence[str] | None = None,
) -> Mapping[str, Path]:
    """Ensure that required pipeline directories exist and return their paths."""

    base_path = io.ensure_dir(base_dir)
    directories: MutableMapping[str, Path] = {}

    def _ensure(relative: str) -> None:
        relative_path = Path(relative)
        target = io.ensure_dir(base_path / relative_path)
        directories[relative_path.as_posix()] = target

    for relative in DEFAULT_PIPELINE_DIRECTORIES:
        _ensure(relative)

    if extra_dirs:
        for relative in extra_dirs:
            _ensure(str(relative))

    return directories


__all__ = [
    "DEFAULT_PIPELINE_DIRECTORIES",
    "initialize_pipeline_directories",
]
