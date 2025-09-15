
"""Environment loading helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

LOGGER = logging.getLogger("utils.config")


class EnvLoadError(RuntimeError):
    """Raised when environment files cannot be processed."""


class MissingSecretError(RuntimeError):
    """Raised when a required configuration value is absent."""


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env(
    *,
    root: Path | None = None,
    env_file: Path | None = None,
    override: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load environment variables from a .env file and process overrides."""

    root_path = root or _default_root()
    env_path = env_file or (root_path / ".env")
    data: dict[str, str] = {}

    try:
        if env_path.exists():
            data.update({k: v for k, v in dotenv_values(str(env_path)).items() if v is not None})
    except Exception as exc:
        LOGGER.error("failed to read env file %s: %s", env_path, exc)
        raise EnvLoadError(f"unable to read env file: {env_path}") from exc

    data.update(os.environ)
    if override:
        data.update(override)
    return data


def require(env: Mapping[str, str], key: str, hint: str = "") -> str:
    value = env.get(key)
    if not value:
        message = f"Missing secret {key}. {hint}".strip()
        LOGGER.error(message)
        raise MissingSecretError(message)
    return value
