from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

# Common French legal forms plus a few international variants that often
# appear in company names. We compare against uppercase tokens.
_LEGAL_FORMS: set[str] = {
    "ASS",
    "ASSOCIATION",
    "AUTOENTREPRENEUR",
    "COOP",
    "COOPERATIVE",
    "EARL",
    "EIRL",
    "EI",
    "EURL",
    "GIE",
    "GMBH",
    "INC",
    "LTD",
    "LLC",
    "MICROENTREPRISE",
    "SA",
    "SARL",
    "SARLU",
    "SAS",
    "SASU",
    "SCA",
    "SCEA",
    "SCI",
    "SC",
    "SCM",
    "SCP",
    "SCS",
    "SCOP",
    "SELARL",
    "SELAFA",
    "SELAS",
    "SELCA",
    "SEP",
    "SNC",
}

# Translation table to help normalise punctuation before tokenisation.
_PUNCT_TRANSLATION = str.maketrans(
    {
        ",": " ",
        ";": " ",
        ":": " ",
        "/": " ",
        "\\": " ",
        "&": " ",
        "'": " ",
        "\u2019": " ",
        "\u201c": " ",
        "\u201d": " ",
        "\xab": " ",
        "\xbb": " ",
        "\u2013": "-",
        "\u2014": "-",
        "_": "-",
    }
)

_LEGAL_FORMS_PATTERN = re.compile(
    r"\b("
    r"SASU|SAS|SARLU|SARL|SA|SCI|SCM|SCP|SCOP|SCS|SCA|SCEA|SC|SNC|SELARL|SELAFA|SELAS|SELCA|SEP|"
    r"EURL|EARL|EI|EIRL|GIE|ASSOCIATION|ASS|COOPERATIVE|COOP|"
    r"AUTO[\s-]*ENTREPRENEUR|AUTO[\s-]*ENTREPRISE|MICRO[\s-]*ENTREPRISE|MICRO[\s-]*ENTREPRENEUR|"
    r"GMBH|INC|LTD|LLC"
    r")\b",
    flags=re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9\s-]+")
_SEPARATOR_RE = re.compile(r"[\s\-]+")


def _strip_accents(value: str) -> str:
    """Return ``value`` without diacritic marks."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _iter_tlds(tlds: Iterable[str]) -> Iterable[str]:
    """Yield cleaned TLD strings."""
    for tld in tlds:
        if not tld:
            continue
        clean = str(tld).strip().lower()
        if not clean:
            continue
        clean = clean[1:] if clean.startswith(".") else clean
        if clean:
            yield clean


def normalize_company_name(name: str) -> str:
    """
    Normalise a company name for consistent slug generation.

    The transformation removes accents, converts to uppercase, strips common
    legal forms and replaces whitespace with single hyphens.
    """
    if not name:
        return ""

    ascii_name = _strip_accents(name).upper()
    ascii_name = ascii_name.replace(".", "")
    ascii_name = ascii_name.translate(_PUNCT_TRANSLATION)
    ascii_name = _LEGAL_FORMS_PATTERN.sub(" ", ascii_name)
    ascii_name = _NON_ALNUM_RE.sub(" ", ascii_name)

    tokens = [
        token
        for token in _SEPARATOR_RE.split(ascii_name)
        if token and token not in _LEGAL_FORMS
    ]

    return "-".join(tokens)


def generate_domain_candidates(name: str, tlds: List[str]) -> List[str]:
    """
    Build candidate domain names from a company name and a collection of TLDs.
    """
    normalized = normalize_company_name(name)
    if not normalized:
        return []

    base_slug = normalized.lower()
    variants: List[str] = []

    def _add_variant(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    _add_variant(base_slug)
    _add_variant(base_slug.replace("-", ""))

    domains: List[str] = []
    for tld in dict.fromkeys(_iter_tlds(tlds)):  # preserves order, removes duplicates
        for label in variants:
            candidate = f"{label}.{tld}"
            if candidate not in domains:
                domains.append(candidate)

    return domains


__all__ = ["normalize_company_name", "generate_domain_candidates"]
