import pandas as pd
import pytest

import enrich.enrich_contacts as contacts_module


class DummyHttpClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self._headers = {"User-Agent": "dummy"}

    def pick_user_agent(self) -> str:
        return self._headers["User-Agent"]

    def close(self) -> None:  # pragma: no cover - stub
        pass

    async def aclose(self) -> None:  # pragma: no cover - stub
        return None


@pytest.fixture(autouse=True)
def patch_http_client(monkeypatch):
    monkeypatch.setattr(contacts_module, "HttpClient", DummyHttpClient)
    monkeypatch.setattr(contacts_module.robots, "configure", lambda *args, **kwargs: None)
    monkeypatch.setattr(contacts_module.robots, "is_allowed", lambda *args, **kwargs: True)
    yield


def make_dataframe():
    return pd.DataFrame([{"site_web": "https://example.com", "ville": "Paris"}])


def test_llm_not_used_when_flag_disabled(monkeypatch):
    called = False

    def llm_stub(html, hints):
        nonlocal called
        called = True
        return {"emails": ["stub@example.com"]}

    def failing_parser(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(contacts_module, "_discover_candidate_urls", lambda *args, **kwargs: ["https://example.com/contact"])
    monkeypatch.setattr(contacts_module, "_fetch_pages", lambda *_args, **_kwargs: {"https://example.com/contact": (200, "<html></html>")})
    monkeypatch.setattr(contacts_module, "_extract_contacts_from_html", failing_parser)
    monkeypatch.setattr(contacts_module, "extract_contacts_llm", llm_stub)

    df_out, summary = contacts_module._process_contacts_serial(make_dataframe(), {"ai": {"fallback_extraction": False}})

    assert called is False
    assert summary["emails_found"] == 0
    assert pd.isna(df_out.loc[0, "email"])


def test_llm_used_when_flag_enabled(monkeypatch):
    received_hints = {}

    def llm_stub(_html, hints):
        nonlocal received_hints
        received_hints = dict(hints)
        return {
            "emails": ["ai@example.com"],
            "phones": ["+33 1 23 45 67 89"],
            "linkedin": ["https://www.linkedin.com/in/example"]
        }

    def failing_parser(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(contacts_module, "_discover_candidate_urls", lambda *args, **kwargs: ["https://example.com/contact"])
    monkeypatch.setattr(contacts_module, "_fetch_pages", lambda *_args, **_kwargs: {"https://example.com/contact": (200, "<html></html>")})
    monkeypatch.setattr(contacts_module, "_extract_contacts_from_html", failing_parser)
    monkeypatch.setattr(contacts_module, "extract_contacts_llm", llm_stub)

    df_out, summary = contacts_module._process_contacts_serial(make_dataframe(), {"ai": {"fallback_extraction": True}})

    assert received_hints["url"] == "https://example.com/contact"
    assert received_hints["base_url"].startswith("https://example.com")
    assert summary["emails_found"] == 1
    assert summary["phones_found"] == 1
    assert df_out.loc[0, "email"] == "ai@example.com"
    assert df_out.loc[0, "telephone"].startswith("+33")
