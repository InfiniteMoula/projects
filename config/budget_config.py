"""Utilities to load budget thresholds from environment or YAML configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from utils import config as env_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetThresholds:
    """Container for common budget thresholds."""

    max_http_requests: int | None = None
    max_http_bytes: int | None = None
    time_budget_min: int | None = None
    ram_mb: int | None = None

    def as_dict(self) -> dict[str, int]:
        """Return thresholds as a plain dictionary ignoring unset values."""

        data: dict[str, int] = {}
        if self.max_http_requests is not None:
            data["max_http_requests"] = self.max_http_requests
        if self.max_http_bytes is not None:
            data["max_http_bytes"] = self.max_http_bytes
        if self.time_budget_min is not None:
            data["time_budget_min"] = self.time_budget_min
        if self.ram_mb is not None:
            data["ram_mb"] = self.ram_mb
        return data


_DEFAULT_BUDGET_FILE = Path(__file__).with_name("budgets.yaml")
_ENV_MAP = {
    "max_http_requests": "BUDGET_MAX_HTTP_REQUESTS",
    "max_http_bytes": "BUDGET_MAX_HTTP_BYTES",
    "time_budget_min": "BUDGET_TIME_MINUTES",
    "ram_mb": "BUDGET_RAM_MB",
}


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return int(float(candidate))
        except ValueError:
            return None
    return None


def _load_yaml_defaults(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("failed to parse budget config %s: %s", path, exc)
        return {}
    if not isinstance(loaded, Mapping):
        LOGGER.warning("budget config %s did not contain a mapping", path)
        return {}
    defaults = loaded.get("defaults", loaded)
    if not isinstance(defaults, Mapping):
        LOGGER.warning("budget defaults in %s are not a mapping", path)
        return {}
    return defaults


def load_budget_thresholds(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
    yaml_path: Path | None = None,
) -> BudgetThresholds:
    """Load budget thresholds from YAML and environment overrides."""

    env_data = env or env_config.load_env(root=root)
    config_path = yaml_path or _DEFAULT_BUDGET_FILE
    yaml_defaults = _load_yaml_defaults(config_path)

    resolved: MutableMapping[str, int | None] = {}
    for key, env_key in _ENV_MAP.items():
        env_value = _coerce_int(env_data.get(env_key))
        if env_value is not None:
            resolved[key] = env_value
        else:
            resolved[key] = _coerce_int(yaml_defaults.get(key))

    return BudgetThresholds(
        max_http_requests=resolved.get("max_http_requests"),
        max_http_bytes=resolved.get("max_http_bytes"),
        time_budget_min=resolved.get("time_budget_min"),
        ram_mb=resolved.get("ram_mb"),
    )


@lru_cache(maxsize=1)
def get_budget_thresholds() -> BudgetThresholds:
    """Return cached budget thresholds."""

    return load_budget_thresholds()
