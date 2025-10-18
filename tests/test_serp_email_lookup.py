from __future__ import annotations

import pytest

from serp import Result, search_emails_via_serp
from serp.providers import DuckDuckGoProvider


@pytest.fixture(autouse=True)
def patch_serp(monkeypatch):
    def fake_search(self, query):  # noqa: D401 - simple stub
        return [
            Result(
                url="https://example.com/contact",
                domain="example.com",
                title="Contact",
                snippet="Contactez-nous via sales@example.com ou support@example.com.",
                rank=1,
            ),
            Result(
                url="https://example.org/info",
                domain="example.org",
                title="Info",
                snippet="Email: HR@EXAMPLE.com pour toute question.",
                rank=2,
            ),
        ]

    original_search = DuckDuckGoProvider.search
    monkeypatch.setattr(DuckDuckGoProvider, "search", fake_search)
    yield
    monkeypatch.setattr(DuckDuckGoProvider, "search", original_search)


def test_search_emails_via_serp_returns_deduplicated_emails():
    emails = search_emails_via_serp("example email", max_results=5)
    assert emails == ["sales@example.com", "support@example.com", "HR@EXAMPLE.com"]


def test_unknown_provider_raises_error():
    with pytest.raises(ValueError):
        search_emails_via_serp("example", provider="unknown")
