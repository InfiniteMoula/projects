from enrich.enrich_contacts import (
    NumberType,
    _extract_emails,
    _extract_phones,
    _normalize_text,
)
from utils.scoring import score_email


def test_extract_emails_deobfuscates_and_marks_domain():
    text = "Contact : Jean [dot] Dupont [at] Example.com et support@example.org"
    emails = _extract_emails(text, "example.com")
    values = {email for email, *_ in emails}
    assert "jean.dupont@example.com" in values
    assert "support@example.org" in values
    for email, on_domain, is_nominative in emails:
        if email == "jean.dupont@example.com":
            assert on_domain is True
            assert is_nominative is True
        if email == "support@example.org":
            assert on_domain is False
            assert is_nominative is False


def test_extract_phones_returns_e164_and_city_match():
    text = "Siège : Paris - Tel 01 23 45 67 89, Portable : +33 (6) 12 34 56 78"
    norm_city = _normalize_text("Paris")
    phones = _extract_phones(text, norm_city)
    phone_map = {number: (ptype, city) for number, ptype, city in phones}
    assert "+33123456789" in phone_map
    assert "+33612345678" in phone_map
    assert phone_map["+33123456789"][0] == NumberType.FIXED_LINE
    assert phone_map["+33612345678"][0] == NumberType.MOBILE


def test_score_email_prefers_nominative_contact():
    score_on_domain = score_email("jean.dupont@example.com", "example.com")
    score_generic = score_email("contact@example.com", "example.com")

    assert score_on_domain > score_generic
