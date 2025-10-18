from __future__ import annotations

"""
Feature engineering utilities for domain prediction.

This module converts company descriptors and candidate domains into numerical
features suitable for training ML models that rank the most plausible website.
"""

import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # Optional dependency used for robust domain parsing.
    import tldextract
except Exception:  # pragma: no cover - optional dependency
    tldextract = None  # type: ignore

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

Candidate = Union[str, Mapping[str, Any]]

_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]{1,63}\.)+[a-z]{2,63}$")
_MULTI_DIGIT_RE = re.compile(r"\d{3,}")
_LEGAL_SUFFIXES = {"fr", "com", "net", "org", "io", "app", "co", "dev", "tech", "biz", "info", "eu"}
_COUNTRY_SUFFIXES = {"fr", "be", "ch", "de", "es", "it", "uk", "us", "ca", "lu"}

_EXTRACT_KEYS = (
    "domain",
    "url",
    "site",
    "homepage",
    "value",
    "candidate",
    "website",
)


@dataclass
class DomainFeatureConfig:
    """Configuration for domain feature extraction."""

    ngram_range: Tuple[int, int] = (3, 5)
    analyzer: str = "char_wb"
    lowercase: bool = True
    max_features: Optional[int] = 4096
    use_idf: bool = True
    enable_mx_lookup: bool = True

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "ngram_range": self.ngram_range,
            "analyzer": self.analyzer,
            "lowercase": self.lowercase,
            "max_features": self.max_features,
            "use_idf": self.use_idf,
        }


def _strip_accents(value: str) -> str:
    """Return ``value`` without diacritic marks."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value)
    return str(value).strip()


def normalize_domain(value: Any) -> str:
    """
    Normalise a raw domain or URL to a hostname without scheme, port or path.
    """
    text = _clean_text(value)
    if not text:
        return ""
    candidate = text
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    candidate = candidate.replace("\\", "/")
    candidate = candidate.split("#", 1)[0].split("?", 1)[0]
    if tldextract:
        try:
            extracted = tldextract.extract(candidate)
        except Exception:
            extracted = None
        if extracted and extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        if extracted and extracted.registered_domain:
            return extracted.registered_domain.lower()
    candidate = re.sub(r"^[a-z]+://", "", candidate, flags=re.I)
    candidate = candidate.split("/", 1)[0]
    candidate = candidate.split("@")[-1]
    candidate = candidate.split(":", 1)[0]
    candidate = candidate.strip().strip(".").lower()
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def _extract_host(value: Any) -> Tuple[str, str, str, int]:
    """
    Return host components for ``value``.

    Returns ``(host, registered_domain, suffix, subdomain_depth)``.
    """
    host = normalize_domain(value)
    if not host:
        return "", "", "", 0
    if tldextract:
        try:
            ext = tldextract.extract(host if "://" not in host else f"https://{host}")
        except Exception:
            ext = None
        if ext:
            registered = ext.registered_domain.lower() if ext.registered_domain else ""
            suffix = ext.suffix.lower() if ext.suffix else ""
            subdomain = ext.subdomain.lower() if ext.subdomain else ""
            depth = len([p for p in subdomain.split(".") if p])
            return host, registered or host, suffix, depth
    parts = [p for p in host.split(".") if p]
    suffix = ""
    registered = host
    if len(parts) >= 2:
        suffix = parts[-1]
        registered = ".".join(parts[-2:])
    depth = max(0, len(parts) - 2)
    return host, registered, suffix, depth


def _domain_core(host: str, registered: str) -> str:
    if not registered:
        return host
    parts = registered.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return registered


def _candidate_score(meta: Mapping[str, Any]) -> Tuple[float, float]:
    raw_score = meta.get("score")
    raw_rank = meta.get("rank", meta.get("position"))
    try:
        score = float(raw_score)
    except Exception:
        score = 0.0
    try:
        rank = float(raw_rank)
    except Exception:
        rank = 0.0
    return score, rank


def _candidate_source(meta: Mapping[str, Any]) -> str:
    raw = meta.get("source") or meta.get("origin") or ""
    return _clean_text(raw).lower()


def _extract_candidate(candidate: Candidate) -> Tuple[str, Dict[str, Any]]:
    if isinstance(candidate, Mapping):
        for key in _EXTRACT_KEYS:
            if key in candidate:
                domain = normalize_domain(candidate[key])
                if domain:
                    score, rank = _candidate_score(candidate)
                    return domain, {
                        "score": score,
                        "rank": rank,
                        "source": _candidate_source(candidate),
                    }
        if "domain" in candidate:
            domain = normalize_domain(candidate["domain"])
            if domain:
                score, rank = _candidate_score(candidate)
                return domain, {
                    "score": score,
                    "rank": rank,
                    "source": _candidate_source(candidate),
                }
        return "", {}
    domain = normalize_domain(candidate)
    if not domain:
        return "", {}
    return domain, {"score": 0.0, "rank": 0.0, "source": ""}


def _normalize_text_feature(value: str) -> str:
    clean = _clean_text(value)
    clean = _strip_accents(clean).lower()
    clean = re.sub(r"[^a-z0-9\s-]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _iter_email_domains(emails: Iterable[Any]) -> List[str]:
    domains: List[str] = []
    for email in emails:
        text = _clean_text(email)
        if not text or "@" not in text:
            continue
        host = normalize_domain(text.split("@", 1)[1])
        if host and host not in domains:
            domains.append(host)
    return domains


try:  # pragma: no cover - optional dependency for DNS MX lookups
    import dns.resolver as dns_resolver  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dns_resolver = None  # type: ignore


@lru_cache(maxsize=2048)
def _has_mx_record(domain: str) -> float:
    """Return 1.0 if the domain seems to have an MX record, else 0.0."""
    if not domain:
        return 0.0
    if dns_resolver is None:
        return 0.0
    try:
        answers = dns_resolver.resolve(domain, "MX", lifetime=1.5)
    except Exception:
        return 0.0
    return 1.0 if answers else 0.0


def _fuzzy_ratio(left: str, right: str, *, scorer: str = "ratio") -> float:
    if not left or not right or fuzz is None:
        return 0.0
    try:
        if scorer == "partial":
            score = fuzz.partial_ratio(left, right)
        elif scorer == "token":
            score = fuzz.token_set_ratio(left, right)
        else:
            score = fuzz.ratio(left, right)
    except Exception:
        return 0.0
    return float(score) / 100.0


def _cosine_similarity(vectorizer: Optional[TfidfVectorizer], left: str, right: str) -> float:
    if not vectorizer or not left or not right:
        return 0.0
    try:
        matrix = vectorizer.transform([left, right])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("cosine similarity failed: %s", exc)
        return 0.0
    return float(sim)


def _naf_tokens(naf: str) -> List[str]:
    naf_clean = _clean_text(naf).upper()
    if not naf_clean:
        return []
    tokens = re.findall(r"[A-Z]+|\d+", naf_clean)
    return [token.lower() for token in tokens if token]


def _postal_prefix(code: str) -> str:
    text = _clean_text(code)
    if not text:
        return ""
    match = re.search(r"\b(\d{2})", text)
    return match.group(1) if match else ""


class DomainFeatureExtractor:
    """Build feature representations for candidate domains."""

    def __init__(
        self,
        config: Optional[DomainFeatureConfig] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.config = config or DomainFeatureConfig()
        self.vectorizer = vectorizer

    def _ensure_vectorizer(self) -> None:
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(**self.config.to_kwargs())

    def fit_vectorizer(self, df: pd.DataFrame, *, name_column: str = "denomination") -> None:
        """
        Fit the TF-IDF vectorizer on company names and known domains.
        """
        self._ensure_vectorizer()
        corpus: List[str] = []

        def _add(value: Any) -> None:
            text = _normalize_text_feature(_clean_text(value))
            if text:
                corpus.append(text)

        if name_column in df.columns:
            for value in df[name_column]:
                _add(value)
        for column in ("denomination", "denomination_usuelle", "raison_sociale", "enseigne", "company_name"):
            if column in df.columns and column != name_column:
                for value in df[column]:
                    _add(value)
        if "domain_true" in df.columns:
            for value in df["domain_true"]:
                _add(value)
        if "candidates" in df.columns:
            for items in df["candidates"]:
                for domain in coerce_candidates(items):
                    _add(domain)
        if not corpus:
            return
        try:
            self.vectorizer.fit(corpus)
        except ValueError:
            log.debug("Skipping vectorizer fit: empty corpus")

    def transform(
        self,
        *,
        name: Any,
        naf: Any,
        city: Any,
        candidates: Optional[Sequence[Candidate]],
        emails: Optional[Sequence[Any]] = None,
        postal_code: Any = None,
    ) -> pd.DataFrame:
        """
        Convert candidate domains into a feature DataFrame for a single company.
        """
        if not candidates:
            return pd.DataFrame(columns=["domain", "normalized_domain"])

        normalized_name = _normalize_text_feature(name)
        naf_parts = _naf_tokens(_clean_text(naf))
        city_text = _normalize_text_feature(city)
        postal_prefix = _postal_prefix(_clean_text(postal_code))
        email_domains = _iter_email_domains(emails or [])

        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            domain, meta = _extract_candidate(candidate)
            if not domain:
                continue
            if domain in seen:
                continue
            seen.add(domain)
            host, registered, suffix, depth = _extract_host(domain)
            if not registered:
                continue
            core = _domain_core(host, registered)
            domain_text = _normalize_text_feature(core.replace("-", " "))
            score, rank = meta.get("score", 0.0), meta.get("rank", 0.0)
            source = meta.get("source", "")

            # Textual similarities
            name_ratio = _fuzzy_ratio(normalized_name, domain_text)
            partial_ratio = _fuzzy_ratio(normalized_name, domain_text, scorer="partial")
            token_ratio = _fuzzy_ratio(normalized_name, domain_text, scorer="token")
            cosine = _cosine_similarity(self.vectorizer, normalized_name, domain_text)

            naf_ratio = 0.0
            naf_exact = 0.0
            if naf_parts and core:
                matches = sum(1 for token in naf_parts if token in core)
                naf_ratio = matches / len(naf_parts)
                naf_exact = 1.0 if any(token == core for token in naf_parts) else 0.0

            city_ratio = _fuzzy_ratio(city_text, domain_text, scorer="partial")
            city_contains = 1.0 if city_text and city_text.replace(" ", "") in host.replace("-", "") else 0.0

            postal_match = 1.0 if postal_prefix and postal_prefix in core else 0.0

            email_exact = 1.0 if registered in email_domains else 0.0
            email_suffix = 1.0 if any(domain.endswith(ed) for ed in email_domains) else 0.0

            mx_score = _has_mx_record(registered) if self.config.enable_mx_lookup else 0.0
            valid_pattern = 1.0 if _DOMAIN_RE.match(registered) else 0.0
            multi_digit = 1.0 if _MULTI_DIGIT_RE.search(registered) else 0.0
            contains_digit = 1.0 if any(char.isdigit() for char in registered) else 0.0
            hyphen_ratio = registered.count("-") / max(1, len(registered))
            digit_ratio = sum(ch.isdigit() for ch in registered) / max(1, len(registered))
            long_label = 1.0 if any(len(part) > 24 for part in registered.split(".")) else 0.0
            suffix_common = 1.0 if suffix in _LEGAL_SUFFIXES else 0.0
            suffix_country = 1.0 if suffix in _COUNTRY_SUFFIXES else 0.0
            has_subdomain = 1.0 if depth > 0 else 0.0

            row = {
                "domain": host,
                "normalized_domain": registered,
                "core_domain": core,
                "suffix": suffix,
                "candidate_index": float(index),
                "candidate_score": float(score),
                "candidate_rank": float(rank),
                "source_semantic": 1.0 if "semantic" in source else 0.0,
                "source_serp": 1.0 if "serp" in source else 0.0,
                "name_ratio": name_ratio,
                "name_partial_ratio": partial_ratio,
                "name_token_ratio": token_ratio,
                "name_cosine": cosine,
                "naf_ratio": naf_ratio,
                "naf_exact": naf_exact,
                "city_ratio": city_ratio,
                "city_contains": city_contains,
                "postal_match": postal_match,
                "email_exact_match": email_exact,
                "email_suffix_match": email_suffix,
                "mx_score": mx_score,
                "valid_pattern": valid_pattern,
                "multi_digit": multi_digit,
                "contains_digit": contains_digit,
                "hyphen_ratio": hyphen_ratio,
                "digit_ratio": digit_ratio,
                "long_label": long_label,
                "suffix_common": suffix_common,
                "suffix_country": suffix_country,
                "has_subdomain": has_subdomain,
                "subdomain_depth": float(depth),
                "domain_length": float(len(registered)),
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["domain", "normalized_domain"])

        frame = pd.DataFrame(rows)
        numeric_cols = [col for col in frame.columns if col not in {"domain", "normalized_domain", "core_domain", "suffix"}]
        frame[numeric_cols] = frame[numeric_cols].fillna(0.0).astype(float)
        return frame

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "vectorizer": self.vectorizer,
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "DomainFeatureExtractor":
        config_data = dict(artifact.get("config") or {})
        config = DomainFeatureConfig(**config_data) if config_data else DomainFeatureConfig()
        vectorizer = artifact.get("vectorizer")
        return cls(config=config, vectorizer=vectorizer)


def coerce_candidates(raw: Any) -> List[Candidate]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    if isinstance(raw, str):
        try:
            value = raw.strip()
            if value.startswith("[") and value.endswith("]"):
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            return [raw]
        return [raw]
    return [raw]


__all__ = [
    "DomainFeatureConfig",
    "DomainFeatureExtractor",
    "coerce_candidates",
    "normalize_domain",
]
