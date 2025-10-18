"""Search engine result page helpers."""

from .email_lookup import search_emails_via_serp
from .providers import BingProvider, DuckDuckGoProvider, Result, SerpProvider

__all__ = [
    "search_emails_via_serp",
    "BingProvider",
    "DuckDuckGoProvider",
    "Result",
    "SerpProvider",
]
