from typing import Iterable, List, Tuple

import dns.resolver

from quality import validation as quality_validation
from utils import email_validation
from utils.email_validation import SMTPVerificationResult, SMTPVerificationStatus


def _fake_mx_records(_: str, __: float) -> List[Tuple[int, str]]:
    return [(10, "mx.test.local")]


def _make_fake_smtp(rcpt_script: Iterable[Tuple[int, bytes]]):
    """Create a fake SMTP class with scripted RCPT responses."""

    script = list(rcpt_script)

    class FakeSMTP:
        def __init__(self, host, port, local_hostname=None, timeout=None):
            assert host == "mx.test.local"
            self._script = list(script)
            self.host = host
            self.port = port
            self.timeout = timeout

        def set_debuglevel(self, level):
            return level

        def ehlo_or_helo_if_needed(self):
            return (250, b"hello")

        def has_extn(self, name: str) -> bool:
            return False

        def starttls(self, context=None):
            return (220, b"ready")

        def mail(self, address: str):
            return (250, b"ok")

        def rcpt(self, address: str):
            if not self._script:
                raise AssertionError(f"Unexpected RCPT TO call for {address}")
            code, message = self._script.pop(0)
            return code, message

        def rset(self):
            return (250, b"reset")

        def quit(self):
            return (221, b"bye")

    return FakeSMTP


def test_passive_smtp_verify_reports_invalid_format():
    result = email_validation.passive_smtp_verify("not-an-email")
    assert result.status is SMTPVerificationStatus.INVALIDE
    assert result.reason == "format"


def test_passive_smtp_verify_handles_missing_mx(monkeypatch):
    def _raise_nxdomain(domain: str, timeout: float):
        raise dns.resolver.NXDOMAIN()

    monkeypatch.setattr(email_validation, "_resolve_mx_records", _raise_nxdomain)
    result = email_validation.passive_smtp_verify("user@example.com")
    assert result.status is SMTPVerificationStatus.INVALIDE
    assert result.reason == "mx_not_found"


def test_passive_smtp_verify_accepts_valid_mailbox(monkeypatch):
    fake_smtp = _make_fake_smtp([(250, b"user ok"), (550, b"no such user")])
    monkeypatch.setattr(email_validation, "_resolve_mx_records", _fake_mx_records)
    monkeypatch.setattr(email_validation.smtplib, "SMTP", fake_smtp)

    result = email_validation.passive_smtp_verify("user@example.com")
    assert result.status is SMTPVerificationStatus.VALIDE
    assert result.catch_all is False
    assert result.mx == "mx.test.local"


def test_passive_smtp_verify_detects_catch_all(monkeypatch):
    fake_smtp = _make_fake_smtp([(250, b"user ok"), (250, b"accepted anyway")])
    monkeypatch.setattr(email_validation, "_resolve_mx_records", _fake_mx_records)
    monkeypatch.setattr(email_validation.smtplib, "SMTP", fake_smtp)

    result = email_validation.passive_smtp_verify("user@example.com")
    assert result.status is SMTPVerificationStatus.DOUTEUX
    assert result.catch_all is True
    assert result.reason == "catch_all"


def test_passive_smtp_verify_rejects_mailbox(monkeypatch):
    fake_smtp = _make_fake_smtp([(550, b"user unknown")])
    monkeypatch.setattr(email_validation, "_resolve_mx_records", _fake_mx_records)
    monkeypatch.setattr(email_validation.smtplib, "SMTP", fake_smtp)

    result = email_validation.passive_smtp_verify("user@example.com")
    assert result.status is SMTPVerificationStatus.INVALIDE
    assert result.reason == "rcpt_denied"
    assert result.code == 550


def test_validate_email_includes_smtp_details(monkeypatch):
    class DummyMX:
        preference = 10
        exchange = "mx.test.local."

    monkeypatch.setattr(
        quality_validation.dns.resolver,
        "resolve",
        lambda domain, record: [DummyMX()],
    )

    fake_result = SMTPVerificationResult(
        email="user@example.com",
        domain="example.com",
        status=SMTPVerificationStatus.INVALIDE,
        reason="rcpt_denied",
        mx="mx.test.local",
        code=550,
        message="user unknown",
        catch_all=False,
    )

    def _fake_passive(email: str, *, timeout: float = 10.0, helo_hostname=None, mail_from=None):
        return fake_result

    monkeypatch.setattr(quality_validation, "passive_smtp_verify", _fake_passive)

    outcome = quality_validation.validate_email(
        "user@example.com",
        check_mx=True,
        check_smtp=True,
    )

    assert not outcome.is_valid
    assert outcome.reason == "rcpt_denied"
    assert outcome.details is not None
    assert outcome.details.get("smtp_status") == "invalide"
    assert outcome.details.get("smtp_code") == 550
