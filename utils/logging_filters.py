"""Reusable logging filters for masking sensitive information."""
from __future__ import annotations

import logging
import re
from typing import Any

from compliance import gdpr

_MASK = "[REDACTED]"

_SECRET_PATTERNS = (
    re.compile(r"(?i)(HUNTER_API_KEY\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(APIFY_API_TOKEN\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(BEARER\s+)([A-Za-z0-9._-]+)"),
    re.compile(r"(?i)(PROXY_URL\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(HTTP_PROXY\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(HTTPS_PROXY\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(API[_-]?TOKEN\s*[=:]\s*)([^\s]+)"),
    re.compile(r"(?i)(API[_-]?KEY\s*[=:]\s*)([^\s]+)"),
)


def _sanitize_string(value: str) -> str:
    if not value:
        return value
    masked = value
    for pattern in _SECRET_PATTERNS:
        masked = pattern.sub(lambda m: f"{m.group(1)}{_MASK}", masked)
    masked = gdpr.anonymize_text(masked)
    return masked


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts API keys and anonymises personal data."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - stdlib signature
        record.msg = _sanitize(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_sanitize(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: _sanitize(val) for key, val in record.args.items()}

        for key, value in list(record.__dict__.items()):
            if isinstance(value, str):
                record.__dict__[key] = _sanitize_string(value)

        if gdpr.should_suppress_record(record):
            return False
        return True


def ensure_global_filter() -> SensitiveDataFilter:
    """Install the sensitive data filter on the root logger."""

    root = logging.getLogger()
    for existing in root.filters:
        if isinstance(existing, SensitiveDataFilter):
            return existing
    filt = SensitiveDataFilter()
    root.addFilter(filt)
    return filt


ensure_global_filter()


__all__ = ["SensitiveDataFilter", "ensure_global_filter"]
