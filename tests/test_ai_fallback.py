import pytest

from enrich import enrich_contacts


class _BoomSoup:
    def __init__(self, *_args, **_kwargs):
        raise ValueError("boom")


def test_html_parse_failure_without_ai(monkeypatch):
    monkeypatch.setattr(enrich_contacts, "BeautifulSoup", _BoomSoup)

    with pytest.raises(ValueError):
        enrich_contacts._extract_contacts_from_html(
            url="https://example.com",
            html="<html></html>",
            company_domain="example.com",
            norm_city="paris",
            fallback_enabled=False,
        )


def test_ai_fallback_extracts_contacts(monkeypatch):
    monkeypatch.setattr(enrich_contacts, "BeautifulSoup", _BoomSoup)

    captured = {}

    def _fake_extract(html, hints):
        captured["html"] = html
        captured["hints"] = dict(hints)
        return {
            "emails": ["Contact@example.com"],
            "phones": ["+33 6 12 34 56 78"],
            "linkedin": ["https://www.linkedin.com/company/example/"],
        }

    monkeypatch.setattr(enrich_contacts, "extract_contacts_with_llm", _fake_extract)

    emails, phones, linkedins = enrich_contacts._extract_contacts_from_html(
        url="https://example.com/contact",
        html="<html>content</html>",
        company_domain="example.com",
        norm_city="paris",
        fallback_enabled=True,
        fallback_hints={"source": "unit-test"},
    )

    assert captured["html"] == "<html>content</html>"
    assert captured["hints"]["url"] == "https://example.com/contact"
    assert captured["hints"]["company_domain"] == "example.com"
    assert captured["hints"]["source"] == "unit-test"

    assert emails == [("contact@example.com", True, False)]
    assert phones and phones[0][0] == "+33612345678"
    assert linkedins == {"https://www.linkedin.com/company/example/"}
