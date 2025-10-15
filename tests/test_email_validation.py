import pytest

from utils import email_validation


@pytest.mark.skipif(email_validation.dns is None, reason="dnspython not installed")
def test_has_mx_record_normalizes_and_caches(monkeypatch):
    email_validation._has_mx_record_normalized.cache_clear()

    calls = []

    def _fake_resolve(domain: str, record_type: str, lifetime: float | None = None):
        calls.append((domain, record_type, lifetime))
        return [object()]

    monkeypatch.setattr(email_validation.dns.resolver, "resolve", _fake_resolve)

    assert email_validation.has_mx_record("Example.COM ") is True
    assert email_validation.has_mx_record("example.com") is True
    assert calls == [("example.com", "MX", email_validation._MX_TIMEOUT)]


@pytest.mark.skipif(email_validation.dns is None, reason="dnspython not installed")
def test_has_mx_record_handles_errors(monkeypatch):
    email_validation._has_mx_record_normalized.cache_clear()

    class _NoAnswer(email_validation.dns.exception.DNSException):
        pass

    def _raise(*_args, **_kwargs):
        raise _NoAnswer()

    monkeypatch.setattr(email_validation.dns.resolver, "resolve", _raise)

    assert email_validation.has_mx_record("invalid.example") is False
