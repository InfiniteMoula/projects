from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

import phonenumbers
from phonenumbers import PhoneNumberType
from phonenumbers.phonenumberutil import NumberParseException
from rapidfuzz import fuzz

try:
    import tldextract
except ImportError:  # pragma: no cover - defensive
    tldextract = None

from constants import (
    GENERIC_EMAIL_DOMAINS as BASE_GENERIC_EMAIL_DOMAINS,
    GENERIC_EMAIL_PREFIXES as BASE_GENERIC_EMAIL_PREFIXES,
)

_LEGAL_FORMS = {
    "SARL",
    "SARLU",
    "SAS",
    "SASU",
    "SA",
    "SCI",
    "SC",
    "SNC",
    "SCS",
    "SCOP",
    "SCA",
    "SCEA",
    "GIE",
    "EURL",
    "EARL",
    "SELARL",
    "EI",
    "EIRL",
    "AUTO ENTREPRISE",
    "AUTO-ENTREPRISE",
    "ASSOCIATION",
    "ASS",
    "COOPERATIVE",
    "COOP",
}

_GENERIC_EMAIL_DOMAINS: Set[str] = {item for item in BASE_GENERIC_EMAIL_DOMAINS if item}

_GENERIC_EMAIL_PREFIXES: Set[str] = {item for item in BASE_GENERIC_EMAIL_PREFIXES if item}

_AREA_GROUPS = {
    "01": {"75", "77", "78", "91", "92", "93", "94", "95"},
    "02": {"02", "14", "18", "22", "27", "28", "29", "35", "36", "37", "39", "41", "44", "45", "49", "50", "53", "56", "58", "60", "61", "76", "79", "80"},
    "03": {"08", "10", "21", "25", "39", "51", "52", "54", "55", "57", "58", "59", "60", "62", "67", "68", "70", "71", "88", "89", "90"},
    "04": {"01", "03", "04", "06", "07", "11", "12", "13", "15", "16", "17", "19", "23", "24", "26", "30", "31", "32", "33", "34", "38", "40", "42", "43", "46", "47", "48", "63", "64", "65", "66", "69", "70", "71", "73", "74", "81", "82", "84", "87"},
    "05": {"09", "11", "12", "15", "16", "17", "19", "23", "24", "31", "32", "33", "40", "46", "47", "48", "63", "64", "65", "66", "79", "81", "82", "86", "87"},
    "09": {"09"},
}

_DOM_AREA_CODES = {
    "971": ("0590", "0591"),
    "972": ("0596",),
    "973": ("0594",),
    "974": ("0262",),
    "975": ("0508",),
    "976": ("0269",),
    "977": ("0590",),
    "978": ("0590",),
    "986": ("0689",),
    "987": ("0089", "0189", "0589", "0989"),
    "988": ("0542", "0568", "0584", "0593"),
}

_DEPARTMENT_AREAS: dict[str, Tuple[str, ...]] = {}
for area_code, departments in _AREA_GROUPS.items():
    prefixed = area_code if area_code.startswith("0") else f"0{area_code}"
    for department in departments:
        existing = _DEPARTMENT_AREAS.get(department, ())
        _DEPARTMENT_AREAS[department] = tuple(sorted(set(existing + (prefixed,))))


def set_generic_email_filters(*, domains: Iterable[str], prefixes: Iterable[str]) -> None:
    """Update the generic email domain/prefix lists used for scoring."""

    normalized_domains: Set[str] = set()
    for domain in domains or []:
        normalized = _extract_registrable_domain(str(domain))
        if normalized:
            normalized_domains.add(normalized.lower())
    if not normalized_domains:
        normalized_domains = {item for item in BASE_GENERIC_EMAIL_DOMAINS if item}

    normalized_prefixes: Set[str] = set()
    for prefix in prefixes or []:
        text = str(prefix).strip().lower()
        if text:
            normalized_prefixes.add(text)
    if not normalized_prefixes:
        normalized_prefixes = {item for item in BASE_GENERIC_EMAIL_PREFIXES if item}

    _GENERIC_EMAIL_DOMAINS.clear()
    _GENERIC_EMAIL_DOMAINS.update(normalized_domains)

    _GENERIC_EMAIL_PREFIXES.clear()
    _GENERIC_EMAIL_PREFIXES.update(normalized_prefixes)


def score_domain(domain: str, company_name: str, city: Optional[str], title: Optional[str]) -> float:
    """
    Compute a confidence score that the provided domain belongs to a company.

    The score is driven by token similarity between the company name and the registrable
    part of the domain, with optional bonuses for full matches and city presence in
    the SERP title/snippet.
    """

    registrable_domain = _extract_registrable_domain(domain)
    if not registrable_domain:
        return 0.0

    company_tokens = _tokenize_company_name(company_name)
    domain_tokens = _tokenize_text(registrable_domain.replace(".", " "))
    if not company_tokens or not domain_tokens:
        return 0.0

    company_text = " ".join(company_tokens)
    domain_text = " ".join(domain_tokens)

    similarity = fuzz.token_set_ratio(company_text, domain_text) / 100.0
    partial = fuzz.partial_ratio(company_text, domain_text) / 100.0
    overlap_ratio = _overlap_ratio(company_tokens, domain_tokens)

    score = (0.55 * similarity) + (0.25 * partial) + (0.2 * overlap_ratio)

    if overlap_ratio >= 1.0 and len(company_tokens) >= 1:
        score += 0.15
    elif overlap_ratio < 0.5:
        score -= 0.1

    if similarity < 0.4:
        score *= 0.75
    if similarity < 0.2:
        score *= 0.5

    normalized_title = _normalize_text(title)
    normalized_city = _normalize_text(city)
    if normalized_city and normalized_title and normalized_city in normalized_title:
        score += 0.08

    generic_penalty = _generic_domain_penalty(domain_tokens)
    score -= generic_penalty

    return _clamp(score)


def score_email(email: str, company_domain: str) -> float:
    """
    Score an email address against a company domain.

    Bonuses:
    - Email domain equals company domain (or a direct subdomain)
    - Local part looks nominative (contains at least one non-generic token)

    Malus:
    - Email domain is generic/free
    """

    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return 0.0

    local_part, domain_part = email.rsplit("@", 1)
    domain_part = domain_part.lower()
    local_part = local_part.strip()

    base_score = 0.2

    normalized_company = _extract_registrable_domain(company_domain)
    normalized_email = _extract_registrable_domain(domain_part)

    if normalized_company:
        if normalized_email == normalized_company:
            base_score += 0.45
        elif normalized_email.endswith("." + normalized_company):
            base_score += 0.35
        elif _same_suffix(normalized_email, normalized_company):
            base_score += 0.25
        else:
            base_score += 0.1
    else:
        base_score += 0.1

    if normalized_email in _GENERIC_EMAIL_DOMAINS:
        base_score -= 0.3

    if _looks_nominative(local_part):
        base_score += 0.25
    elif local_part and local_part.lower() not in _GENERIC_EMAIL_PREFIXES:
        base_score += 0.05

    digit_ratio = _digit_ratio(local_part)
    if digit_ratio >= 0.4:
        base_score -= 0.1

    return _clamp(base_score)


def score_phone(number: str, city_code: Optional[str]) -> float:
    """
    Score a phone number for relevance to a company located in a given department.

    Bonuses:
    - Landline numbers score higher than mobile (malus applied to mobile-only numbers)
    - Matching area code with the department boosts confidence
    """

    number = (number or "").strip()
    if not number:
        return 0.0

    sanitized = _sanitize_number(number)
    if not sanitized:
        return 0.0

    score = 0.25
    is_valid = False
    is_fixed = False
    is_mobile = False

    try:
        parsed = phonenumbers.parse(number, "FR")
        is_valid = phonenumbers.is_valid_number(parsed)
        number_type = phonenumbers.number_type(parsed)
        if number_type == PhoneNumberType.FIXED_LINE:
            is_fixed = True
        elif number_type == PhoneNumberType.MOBILE:
            is_mobile = True
        elif number_type == PhoneNumberType.FIXED_LINE_OR_MOBILE:
            is_fixed = True
            is_mobile = True
    except NumberParseException:
        parsed = None

    if not parsed:
        if sanitized.startswith(("06", "07", "336", "337")):
            is_mobile = True
        elif sanitized.startswith(("01", "02", "03", "04", "05", "09", "080", "081", "082", "089")):
            is_fixed = True

    if is_fixed:
        score += 0.5
    elif is_mobile and not is_fixed:
        score -= 0.1
        score += 0.2
    else:
        score += 0.1

    if is_valid:
        score += 0.05
    elif len(sanitized) < 9:
        score -= 0.1

    department_codes = _candidate_area_codes(city_code)
    if department_codes and _area_code_matches(sanitized, department_codes):
        score += 0.15

    return _clamp(score)


def _extract_registrable_domain(value: Optional[str]) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""

    url_match = re.match(r"^[a-z]+://", value)
    if url_match:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0]

    if tldextract:
        parts = tldextract.extract(value)
        if parts.domain and parts.suffix:
            return f"{parts.domain}.{parts.suffix}".lower()
        if parts.domain:
            return parts.domain.lower()
    value = value.lstrip("www.")
    return value


def _tokenize_company_name(text: Optional[str]) -> Tuple[str, ...]:
    normalized = _normalize_text(text)
    if not normalized:
        return ()
    for form in _LEGAL_FORMS:
        normalized = normalized.replace(_normalize_text(form), " ")
    tokens = tuple(token for token in normalized.split() if token)
    return tokens


def _tokenize_text(text: Optional[str]) -> Tuple[str, ...]:
    normalized = _normalize_text(text)
    if not normalized:
        return ()
    return tuple(token for token in normalized.split() if token)


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    stripped = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    stripped = stripped.lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return round(value, 3)


def _strip_non_digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _sanitize_number(number: str) -> str:
    digits = _strip_non_digits(number)
    if not digits:
        return ""
    if digits.startswith("0033"):
        digits = digits[4:]
    if digits.startswith("33"):
        digits = digits[2:]
    if not digits.startswith("0"):
        digits = "0" + digits
    return digits


def _candidate_area_codes(city_code: Optional[str]) -> Tuple[str, ...]:
    if not city_code:
        return ()
    code = str(city_code).strip().upper()
    if not code:
        return ()

    if code in {"2A", "2B"}:
        code_key = "20"
    else:
        digits = re.findall(r"\d+", code)
        if not digits:
            return ()
        numeric = digits[0]
        if numeric.startswith("97") or numeric.startswith("98"):
            key = numeric[:3]
            return _DOM_AREA_CODES.get(key, ())
        code_key = numeric[:2]

    return _DEPARTMENT_AREAS.get(code_key, ())


def _area_code_matches(number: str, area_codes: Sequence[str]) -> bool:
    for area in area_codes:
        if not area:
            continue
        normalized = area if area.startswith("0") else f"0{area}"
        if number.startswith(normalized):
            return True
    return False


def _overlap_ratio(company_tokens: Sequence[str], domain_tokens: Sequence[str]) -> float:
    if not company_tokens:
        return 0.0
    company_set: Set[str] = set(company_tokens)
    domain_set: Set[str] = set(domain_tokens)
    if not company_set or not domain_set:
        return 0.0
    shared = company_set & domain_set
    return len(shared) / max(1, len(company_set))


def _generic_domain_penalty(tokens: Sequence[str]) -> float:
    GENERIC = {"contact", "support", "blog", "site", "web", "group", "digital"}
    if any(token in GENERIC for token in tokens):
        return 0.05
    return 0.0


def _looks_nominative(local_part: str) -> bool:
    if not local_part:
        return False
    tokens = [chunk for chunk in re.split(r"[.\-_]+", local_part.lower()) if chunk]
    if not tokens:
        return False
    for token in tokens:
        if len(token) >= 3 and token.isalpha() and token not in _GENERIC_EMAIL_PREFIXES:
            return True
    return False


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / len(text)


def score_lead(lead: Mapping[str, Any]) -> int:
    """
    Compute a heuristic engagement score for a lead dictionary.

    The function inspects common fields (email, domain, sector, employee_count,
    etc.) and returns an integer between 0 and 100.
    """

    if not lead:
        return 0

    score = 50.0

    email = str(lead.get("email") or "").strip()
    website = str(lead.get("domain") or lead.get("website") or lead.get("url") or "").strip()
    company = str(lead.get("company") or lead.get("company_name") or lead.get("name") or "").strip()
    city = str(lead.get("city") or "").strip()
    country = str(lead.get("country") or "").strip()

    email_domain = ""
    if email and "@" in email:
        local_part, _, domain_part = email.rpartition("@")
        email_domain = _extract_registrable_domain(domain_part)
        if email_domain in _GENERIC_EMAIL_DOMAINS:
            score -= 30
        elif email_domain:
            score += 20
        if _looks_nominative(local_part):
            score += 5
        else:
            score -= 5
    elif email:
        score -= 15
    else:
        score -= 20

    company_domain = _extract_registrable_domain(website)
    if company_domain:
        score += 10
        if email_domain and _same_suffix(email_domain, company_domain):
            score += 10
    else:
        score -= 10

    if company:
        score += 5
    else:
        score -= 5

    sector_raw = lead.get("sector") or lead.get("industry") or lead.get("naf") or ""
    sector_text = _normalize_text(str(sector_raw))
    if sector_text:
        high_value = {
            "software",
            "saas",
            "tech",
            "information",
            "industrie",
            "finance",
            "medical",
            "health",
            "healthcare",
            "biotech",
            "aeronautique",
            "aerospace",
            "ingenierie",
            "engineering",
        }
        low_value = {"association", "asso", "etudiant", "student", "ngo", "club"}
        if any(keyword in sector_text for keyword in high_value):
            score += 8
        elif any(keyword in sector_text for keyword in low_value):
            score -= 8
        else:
            score += 2
    else:
        score -= 5

    employee_count = None
    for key in ("employee_count", "employees", "headcount", "staff", "effectif", "effectifs"):
        raw_value = lead.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            employee_count = int(raw_value)
            break
        if isinstance(raw_value, str):
            digits = re.findall(r"\d+", raw_value)
            if digits:
                try:
                    employee_count = int(digits[-1])
                    break
                except ValueError:
                    continue
    if employee_count is not None:
        if employee_count >= 5000:
            score += 15
        elif employee_count >= 500:
            score += 12
        elif employee_count >= 50:
            score += 8
        elif employee_count >= 10:
            score += 4
        elif employee_count <= 1:
            score -= 8
        else:
            score -= 3
    else:
        score -= 3

    revenue = None
    for key in ("revenue", "turnover", "ca", "chiffre_affaires"):
        raw_value = lead.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            revenue = float(raw_value)
            break
        if isinstance(raw_value, str):
            digits = re.findall(r"\d+", raw_value.replace(" ", ""))
            if digits:
                try:
                    revenue = float(digits[0])
                    break
                except ValueError:
                    continue
    if revenue is not None:
        if revenue >= 1_000_000:
            score += 8
        elif revenue >= 100_000:
            score += 4
        else:
            score -= 2

    if lead.get("phone") or lead.get("phone_number"):
        score += 5

    if company_domain and lead.get("linkedin"):
        score += 5

    if city:
        score += 2
    if country:
        score += 2

    completeness_fields = ["address", "zip", "postal_code", "linkedin", "phone", "website"]
    filled = sum(1 for field in completeness_fields if lead.get(field))
    score += filled

    return max(0, min(100, int(round(score))))


def _same_suffix(domain_a: str, domain_b: str) -> bool:
    if not domain_a or not domain_b:
        return False
    parts_a = domain_a.split(".")[-2:]
    parts_b = domain_b.split(".")[-2:]
    return parts_a == parts_b


__all__ = ["score_domain", "score_email", "score_phone", "set_generic_email_filters", "score_lead"]
