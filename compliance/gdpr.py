"""Minimal GDPR helpers for log anonymisation and opt-out handling."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Iterable, Iterator, Sequence

LOGGER = logging.getLogger("compliance.gdpr")

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d().\s-]{6,}\d")

_SALT = os.getenv("GDPR_LOG_SALT", "InfiniteMoula")
_OPT_OUT_RAW: set[str] = set()
_OPT_OUT_HASHED: set[str] = set()


def _normalise_email(value: str) -> str:
    return value.strip().lower()


def _normalise_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return digits


def _normalise_generic(value: str) -> str:
    return value.strip().lower()


def hash_identifier(value: str) -> str:
    """Return a deterministic hash for the provided identifier."""

    data = (value + _SALT).encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()


def anonymize_text(text: str) -> str:
    """Replace emails and phone numbers in ``text`` by hashed placeholders."""

    def _replace_email(match: re.Match[str]) -> str:
        raw = _normalise_email(match.group(0))
        digest = hash_identifier(raw)
        return f"[email:{digest[:12]}]"

    def _replace_phone(match: re.Match[str]) -> str:
        raw = _normalise_phone(match.group(0))
        if not raw:
            return "[phone:unknown]"
        digest = hash_identifier(raw)
        return f"[phone:{digest[:12]}]"

    anonymised = _EMAIL_RE.sub(_replace_email, text)
    anonymised = _PHONE_RE.sub(_replace_phone, anonymised)
    return anonymised


def register_opt_out(identifiers: Iterable[str] | str) -> None:
    """Register identifiers that should be treated as opt-out."""

    if isinstance(identifiers, str):
        values: Iterator[str] = iter([identifiers])
    else:
        values = iter(identifiers)

    for item in values:
        value = item.strip()
        if not value:
            continue
        if value.startswith("hash:"):
            hashed = value[5:]
            if hashed:
                _OPT_OUT_HASHED.add(hashed)
            continue
        generic = _normalise_generic(value)
        if generic:
            _OPT_OUT_RAW.add(generic)
            _OPT_OUT_HASHED.add(hash_identifier(generic))
        digits = _normalise_phone(value)
        if digits and digits not in _OPT_OUT_RAW:
            _OPT_OUT_RAW.add(digits)
            _OPT_OUT_HASHED.add(hash_identifier(digits))


def load_opt_out_from_env() -> None:
    """Populate opt-out identifiers using the ``GDPR_OPT_OUT_IDS`` variable."""

    raw = os.getenv("GDPR_OPT_OUT_IDS", "")
    if not raw:
        return
    register_opt_out(token for token in raw.split(","))


def is_identifier_opted_out(value: str) -> bool:
    """Return ``True`` if ``value`` is registered as opt-out."""

    if not value:
        return False
    normalised = _normalise_generic(value)
    if normalised in _OPT_OUT_RAW:
        return True
    digest = hash_identifier(normalised)
    if digest in _OPT_OUT_HASHED:
        return True
    digits = _normalise_phone(value)
    if digits and digits in _OPT_OUT_RAW:
        return True
    if digits and hash_identifier(digits) in _OPT_OUT_HASHED:
        return True
    return False


def iter_record_identifiers(record: logging.LogRecord) -> Iterator[str]:
    keys: Sequence[str] = (
        "data_subject",
        "subject_id",
        "user_id",
        "email",
        "phone",
    )
    for key in keys:
        value = getattr(record, key, None)
        if not value:
            continue
        if key == "phone":
            yield _normalise_phone(str(value))
        else:
            yield _normalise_generic(str(value))


def should_suppress_record(record: logging.LogRecord) -> bool:
    """Return ``True`` when the log record should be dropped due to opt-out."""

    if getattr(record, "gdpr_opt_out", False):
        return True
    for identifier in iter_record_identifiers(record):
        if not identifier:
            continue
        if identifier in _OPT_OUT_RAW or hash_identifier(identifier) in _OPT_OUT_HASHED:
            return True
    return False


load_opt_out_from_env()

__all__ = [
    "anonymize_text",
    "hash_identifier",
    "is_identifier_opted_out",
    "iter_record_identifiers",
    "register_opt_out",
    "should_suppress_record",
]
