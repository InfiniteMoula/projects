from __future__ import annotations

import re
from typing import Iterable, List, Mapping, MutableMapping, Type

from net.http_client import HttpClient
from serp.providers import BingProvider, DuckDuckGoProvider, Result, SerpProvider

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PROVIDER_FACTORIES: Mapping[str, Type[SerpProvider]] = {
    "duckduckgo": DuckDuckGoProvider,
    "bing": BingProvider,
}


def _build_http_client(cfg: Mapping[str, object] | None) -> HttpClient:
    merged: MutableMapping[str, object] = dict(cfg or {})
    default_headers = dict(_DEFAULT_HEADERS)
    candidate = merged.get("default_headers")
    if isinstance(candidate, Mapping):
        default_headers.update(dict(candidate))
    merged["default_headers"] = default_headers
    return HttpClient(merged)


def _create_provider(
    provider: str,
    *,
    http_client: HttpClient,
    max_results: int,
    extra_config: Mapping[str, object] | None = None,
) -> SerpProvider:
    provider_class = _PROVIDER_FACTORIES.get(provider.lower())
    if provider_class is None:
        known = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ValueError(f"Unknown provider '{provider}'. Known providers: {known}")
    provider_cfg: MutableMapping[str, object] = {"max_results": max_results}
    if extra_config:
        provider_cfg.update(dict(extra_config))
    return provider_class(provider_cfg, http_client)


def _extract_emails(results: Iterable[Result]) -> List[str]:
    found: List[str] = []
    seen = set()
    for item in results:
        snippet = item.snippet or ""
        for match in EMAIL_PATTERN.findall(snippet):
            lowered = match.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            found.append(match)
    return found


def search_emails_via_serp(
    query: str,
    *,
    provider: str = "duckduckgo",
    max_results: int = 10,
    http_config: Mapping[str, object] | None = None,
    provider_config: Mapping[str, object] | None = None,
) -> List[str]:
    """Search for emails in SERP snippets.

    Args:
        query: Search query, e.g. "@example.com" or "company name email".
        provider: Name of the SERP provider ("duckduckgo" or "bing").
        max_results: Maximum number of results fetched from the provider.
        http_config: Optional overrides for the :class:`HttpClient` configuration.
        provider_config: Optional additional configuration forwarded to the provider.

    Returns:
        A list of deduplicated email addresses discovered in visible snippets.
    """

    client = _build_http_client(http_config)
    try:
        serp_provider = _create_provider(
            provider,
            http_client=client,
            max_results=max(1, int(max_results)),
            extra_config=provider_config,
        )
        results = serp_provider.search(query)
        return _extract_emails(results)
    finally:
        client.close()


__all__ = ["search_emails_via_serp"]
