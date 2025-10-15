from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

__all__ = [
    "GENERIC_DOMAINS",
    "GENERIC_EMAIL_DOMAINS",
    "GENERIC_EMAIL_PREFIXES",
    "load_word_list",
]


def _data_dir() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_word_list(filename: str) -> Tuple[str, ...]:
    """Load and normalize a list of newline separated words from *filename*."""

    path = _data_dir() / filename
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return tuple()

    items = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        items.append(stripped.lower())

    # Preserve order while removing duplicates
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


GENERIC_DOMAINS = load_word_list("generic_domains.txt")
GENERIC_EMAIL_DOMAINS = load_word_list("generic_email_domains.txt")
GENERIC_EMAIL_PREFIXES = load_word_list("generic_email_prefixes.txt")
