from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from utils import io

LOGGER = logging.getLogger("config.enrichment_config")


class HttpClientConfig(BaseModel):
    """Settings passed to ``net.http_client.HttpClient``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    timeout: float = Field(default=8.0, ge=0.0)
    max_concurrent_requests: int = Field(default=6, ge=1)
    per_host_limit: int = Field(default=2, ge=1)
    retry_attempts: int = Field(default=3, ge=1)
    backoff_factor: float = Field(default=0.5, ge=0.0, alias="delay_base")
    max_backoff: float = Field(default=10.0, ge=0.0)
    cache_dir: Optional[Path] = None
    cache_ttl_days: float = Field(default=1.0, ge=0.0)
    default_headers: Dict[str, str] = Field(default_factory=dict)
    follow_redirects: bool = True
    respect_robots: bool = True
    robots_cache_ttl: float = Field(default=3600.0, ge=0.0)
    max_connections: Optional[int] = Field(default=None, ge=1)
    max_keepalive_connections: Optional[int] = Field(default=None, ge=1)
    user_agents_file: Optional[Path] = None
    user_agents: List[str] = Field(default_factory=list)
    per_host_delay_seconds: Dict[str, float] = Field(default_factory=dict)
    default_per_host_delay: float = Field(default=0.0, ge=0.0)

    @field_validator("cache_dir", "user_agents_file", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("user_agents", mode="after")
    @classmethod
    def _sanitize_user_agents(cls, value: Iterable[str]) -> List[str]:
        if not value:
            return []
        return [ua.strip() for ua in value if isinstance(ua, str) and ua.strip()]

    @field_validator("per_host_delay_seconds", mode="after")
    @classmethod
    def _normalize_delays(cls, value: Mapping[str, object]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for host, delay in dict(value or {}).items():
            host_key = str(host).strip().lower()
            if not host_key:
                continue
            try:
                normalized[host_key] = max(0.0, float(delay))
            except (TypeError, ValueError):
                normalized[host_key] = 0.0
        return normalized


class SerpProviderConfig(BaseModel):
    """Configuration passed to SERP provider implementations."""

    model_config = ConfigDict(extra="allow")

    max_results: int = Field(default=10, ge=1)
    delay_base: float = Field(default=0.0, ge=0.0)


class DomainsConfig(BaseModel):
    """Domain discovery enrichment settings."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    providers: List[str] = Field(default_factory=lambda: ["bing", "duckduckgo"])
    providers_config: Dict[str, SerpProviderConfig] = Field(default_factory=dict)
    http_client: HttpClientConfig = Field(default_factory=HttpClientConfig)
    serp_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    heuristic_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    heuristic_tlds: List[str] = Field(default_factory=lambda: ["fr", "com", "eu", "net", "org"], alias="tlds")
    heuristic_prefixes: List[str] = Field(default_factory=lambda: ["", "www."], alias="prefixes")
    extra_generic_domains: List[str] = Field(default_factory=list)

    @field_validator("providers", "heuristic_tlds", "heuristic_prefixes", mode="after")
    @classmethod
    def _sanitize_sequences(cls, value: Iterable[str], info: ValidationInfo) -> List[str]:
        if not value:
            return []
        cleaned: List[str] = []
        allow_blank = info.field_name == "heuristic_prefixes"
        for item in value:
            text = str(item).strip()
            if info.field_name == "heuristic_tlds":
                text = text.lstrip(".")
            if text:
                cleaned.append(text)
            elif allow_blank:
                cleaned.append("")
        return cleaned

    @field_validator("extra_generic_domains", mode="after")
    @classmethod
    def _sanitize_generic_domains(cls, value: Iterable[str]) -> List[str]:
        if not value:
            return []
        cleaned: List[str] = []
        for item in value:
            text = str(item).strip().lower()
            if text:
                cleaned.append(text)
        return cleaned


DEFAULT_CONTACT_PATHS = [
    "/contact",
    "/contacts",
    "/nous-contacter",
    "/nous_contacter",
    "/contactez-nous",
    "/contactez_nous",
    "/mentions-legales",
    "/mentions_legales",
    "/mentions",
    "/a-propos",
    "/a_propos",
    "/about",
    "/about-us",
    "/privacy",
    "/politique-de-confidentialite",
    "/politique_confidentialite",
    "/rgpd",
]

DEFAULT_EMAIL_GENERIC_PREFIXES = [
    "contact",
    "info",
    "hello",
    "support",
    "service",
    "commercial",
    "vente",
    "admin",
    "administration",
    "compta",
    "billing",
    "facturation",
    "direction",
    "rh",
    "recrutement",
    "postmaster",
    "noreply",
    "no-reply",
    "bonjour",
]


class ContactsConfig(BaseModel):
    """Contact enrichment settings."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    http_client: HttpClientConfig = Field(default_factory=HttpClientConfig)
    paths: List[str] = Field(default_factory=lambda: DEFAULT_CONTACT_PATHS.copy(), alias="pages_to_scan")
    max_pages_per_site: int = Field(default=8, ge=1)
    sitemap_limit: int = Field(default=5, ge=0)
    use_sitemap: bool = True
    use_robots: bool = True
    email_generic_domains: List[str] = Field(default_factory=list)
    email_generic_prefixes: List[str] = Field(default_factory=lambda: DEFAULT_EMAIL_GENERIC_PREFIXES.copy())

    @field_validator("paths", "email_generic_domains", "email_generic_prefixes", mode="after")
    @classmethod
    def _sanitize_list(cls, value: Iterable[str]) -> List[str]:
        if not value:
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class LinkedinConfig(BaseModel):
    """LinkedIn enrichment settings."""

    model_config = ConfigDict(extra="allow")

    providers: List[str] = Field(default_factory=lambda: ["bing", "duckduckgo"])
    providers_config: Dict[str, SerpProviderConfig] = Field(default_factory=dict)
    http_client: HttpClientConfig = Field(default_factory=HttpClientConfig)

    @field_validator("providers", mode="after")
    @classmethod
    def _sanitize_providers(cls, value: Iterable[str]) -> List[str]:
        if not value:
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class AdaptiveConfig(BaseModel):
    """Configuration controlling adaptive tuning."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    max_ram_gb: float = Field(default=6.0, ge=0.0)
    min_concurrency: int = Field(default=10, ge=1)
    max_concurrency: int = Field(default=60, ge=1)
    min_chunk: int = Field(default=300, ge=1)
    max_chunk: int = Field(default=1200, ge=1)

    @field_validator("max_concurrency")
    @classmethod
    def _ensure_concurrency_bounds(cls, value: int, info: ValidationInfo) -> int:
        minimum = info.data.get("min_concurrency", 1)
        if value < minimum:
            raise ValueError("max_concurrency must be >= min_concurrency")
        return value

    @field_validator("max_chunk")
    @classmethod
    def _ensure_chunk_bounds(cls, value: int, info: ValidationInfo) -> int:
        minimum = info.data.get("min_chunk", 1)
        if value < minimum:
            raise ValueError("max_chunk must be >= min_chunk")
        return value


class EnrichmentConfig(BaseModel):
    """Root configuration for enrichment steps."""

    model_config = ConfigDict(extra="allow")

    use_domains: bool = True
    use_contacts: bool = True
    use_linkedin: bool = True
    domains: DomainsConfig = Field(default_factory=DomainsConfig)
    contacts: ContactsConfig = Field(default_factory=ContactsConfig)
    linkedin: LinkedinConfig = Field(default_factory=LinkedinConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)


@lru_cache(maxsize=1)
def load_enrichment_config(path: str | Path = "config/enrichment.yaml") -> EnrichmentConfig:
    """Load and validate the enrichment configuration from *path*."""

    config_path = Path(path).expanduser()
    if not config_path.exists():
        LOGGER.debug("Enrichment config %s not found; using defaults", config_path)
        return EnrichmentConfig()

    try:
        raw_data = io.read_text(config_path)
    except io.IoError as exc:
        raise RuntimeError(f"Unable to read enrichment config: {config_path}") from exc

    try:
        payload = yaml.safe_load(raw_data) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in enrichment config {config_path}: {exc}") from exc

    try:
        return EnrichmentConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid enrichment config {config_path}: {exc}") from exc


__all__ = [
    "ContactsConfig",
    "AdaptiveConfig",
    "DomainsConfig",
    "EnrichmentConfig",
    "HttpClientConfig",
    "LinkedinConfig",
    "SerpProviderConfig",
    "load_enrichment_config",
]
