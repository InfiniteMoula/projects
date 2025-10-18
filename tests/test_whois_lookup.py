from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from utils import whois_lookup


class DummyWhois(SimpleNamespace):
    def __bool__(self) -> bool:  # emulate truthy behaviour of python-whois response
        return True


@pytest.fixture(autouse=True)
def restore_whois(monkeypatch: pytest.MonkeyPatch) -> None:
    original = whois_lookup.whois

    yield

    monkeypatch.setattr(whois_lookup, "whois", original)


def make_response(**data: Any) -> DummyWhois:
    # python-whois returns an object supporting attribute and dict-style access
    response = DummyWhois(**data)
    response.__dict__.update(data)
    return response


def test_returns_first_public_email(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(emails=["REDACTED@privacy.com", "owner@example.com"])
    monkeypatch.setattr(whois_lookup, "whois", SimpleNamespace(whois=lambda _: response))

    assert whois_lookup.get_public_whois_email("example.com") == "owner@example.com"


def test_returns_none_when_only_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    response = make_response(emails=["redacted@privacyprotect.org", "contact@proxy.example"])
    monkeypatch.setattr(whois_lookup, "whois", SimpleNamespace(whois=lambda _: response))

    assert whois_lookup.get_public_whois_email("example.com") is None


def test_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(_domain: str) -> None:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(whois_lookup, "whois", SimpleNamespace(whois=raise_error))

    assert whois_lookup.get_public_whois_email("example.com") is None
