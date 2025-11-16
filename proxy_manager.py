"""Centralised proxy configuration for outbound HTTP clients.

Example configuration (EXAMPLE — do not commit real credentials):

    # PowerShell
    # set PYPROXY_ENABLED=true
    # set PYPROXY_HOST=925559a762982876.zqq.na.pyproxy.io
    # set PYPROXY_PORT=16666
    # set PYPROXY_USERNAME=deuxt921-zone-resi
    # set PYPROXY_PASSWORD=deuxt129

The same values can be stored in ``config/local_proxy.toml`` which must stay
out of version control. The default structure is::

    [pyproxy]
    enabled = true
    host = "925559a762982876.zqq.na.pyproxy.io"
    port = 16666
    username = "deuxt921-zone-resi"
    password = "deuxt129"

Set ``PROXY_PROVIDER`` to override the provider at runtime without touching
the code (defaults to ``pyproxy``). Each provider reads the matching
``<PROVIDER>_HOST``/``PORT``/``USERNAME``/``PASSWORD`` environment variables.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - py310 fallback
    raise RuntimeError("Python 3.11+ is required for tomllib support") from exc


LOGGER = logging.getLogger("proxy_manager")
DEFAULT_CONFIG_PATH = Path("config/local_proxy.toml")
PROXY_PROVIDER_ENV = "PROXY_PROVIDER"


def load_bool_env(var_name: str, default: bool) -> bool:
    """Parse true/false style environment variables safely."""

    value = os.getenv(var_name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class ProxyManager:
    """Load and expose proxy credentials for HTTP clients."""

    def __init__(
        self,
        provider: str = "pyproxy",
        enabled: bool | None = None,
        *,
        config_path: Path | str | None = None,
    ) -> None:
        """
        Args:
            provider: Name of the proxy provider (mapped to env prefixes).
            enabled: Override switch. If ``None`` the value is read from env/config.
            config_path: Optional custom path to the local proxy config file.
        """

        provider_choice = os.getenv(PROXY_PROVIDER_ENV, provider)
        self.provider = provider_choice.lower()
        self.env_prefix = f"{self.provider.upper()}_"
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self._file_settings = self._load_local_config()
        self._enabled = self._resolve_enabled(enabled)
        self._host = self._get_setting("HOST", "host")
        self._port = self._get_setting("PORT", "port")
        self._username = self._get_setting("USERNAME", "username")
        self._password = self._get_setting("PASSWORD", "password")
        self._log_status()

    @property
    def enabled(self) -> bool:
        """Return True when proxy usage is enabled."""

        return self._enabled

    def _load_local_config(self) -> Dict[str, Dict[str, Any]]:
        if not self.config_path.exists():
            return {}
        try:
            with self.config_path.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load proxy config %s: %s", self.config_path, exc)
            return {}
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items() if isinstance(v, dict)}
        return {}

    def _provider_section(self) -> Dict[str, Any]:
        return dict(self._file_settings.get(self.provider, {}))

    def _resolve_enabled(self, override: bool | None) -> bool:
        if override is not None:
            return bool(override)
        env_toggle_name = f"{self.env_prefix}ENABLED"
        if env_toggle_name in os.environ:
            return load_bool_env(env_toggle_name, True)
        section = self._provider_section()
        enabled_value = section.get("enabled")
        if isinstance(enabled_value, bool):
            return enabled_value
        if isinstance(enabled_value, str):
            return enabled_value.strip().lower() in {"1", "true", "yes", "on"}
        return True

    def _get_setting(self, env_suffix: str, config_key: str) -> Optional[str]:
        env_name = f"{self.env_prefix}{env_suffix}"
        value = os.getenv(env_name)
        if value:
            return value.strip()
        section = self._provider_section()
        candidate = section.get(config_key)
        if candidate is None:
            return None
        return str(candidate).strip()

    def _log_status(self) -> None:
        if self.enabled:
            if self._host and self._port:
                LOGGER.info(
                    "Proxy enabled for %s via %s:%s",
                    self.provider,
                    self._host,
                    self._port,
                )
            else:  # pragma: no cover - informational
                LOGGER.warning("Proxy enabled for %s but host/port missing", self.provider)
        else:
            LOGGER.info("Proxy disabled for provider %s", self.provider)

    def get_proxy_url(self) -> Optional[str]:
        """Return ``http://username:password@host:port`` or ``None`` when disabled."""

        if not self.enabled:
            return None
        if not all([self._username, self._password, self._host, self._port]):
            LOGGER.warning("Proxy configuration incomplete for %s; skipping proxy", self.provider)
            return None
        return f"http://{self._username}:{self._password}@{self._host}:{self._port}"

    def as_requests(self) -> Optional[Dict[str, str]]:
        """Return ``requests`` style proxies mapping or ``None`` when disabled."""

        proxy_url = self.get_proxy_url()
        if not proxy_url:
            return None
        return {"http": proxy_url, "https": proxy_url}

    def as_aiohttp(self) -> Optional[str]:
        """Return proxy URL for aiohttp client usage."""

        return self.get_proxy_url()

    def masked_proxy(self) -> Optional[str]:
        """Return proxy URL with masked password for safe logging."""

        proxy_url = self.get_proxy_url()
        if not proxy_url:
            return None
        safe_password = "***"
        if self._password:
            return proxy_url.replace(self._password, safe_password, 1)
        return proxy_url
