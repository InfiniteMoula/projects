"""Business lead scoring utilities.

This module provides a simple heuristic scoring mechanism that evaluates how
qualified a lead is based on four high-level pillars:

* Contact completeness (email, phone, website, address, LinkedIn)
* Company size (employee headcount and/or revenue information)
* Activity information (quality of the NAF/APE codes or activity labels)
* Website recency (based on the ``date_creation`` field)

Each component is scored between 0 and 1 and then combined into a 0-100
``score_business`` metric.  The heuristics are deliberately transparent so they
can be tuned easily in the future and work with partially filled datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional
import math
import re

try:  # Optional dependency – only required when using pandas helpers.
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is available in tests.
    pd = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LeadScoreWeights:
    """Weights applied to each scoring component.

    The weights are normalized so that their sum equals 1.0.  Consumers may
    provide custom weights; the normalization makes sure the overall score
    always stays in the 0-100 interval.
    """

    contact: float = 0.40
    company_size: float = 0.25
    activity: float = 0.15
    web_recency: float = 0.20

    def normalized(self) -> "LeadScoreWeights":
        total = self.contact + self.company_size + self.activity + self.web_recency
        if total <= 0:
            # Fall back to an even split to avoid a division by zero.
            return LeadScoreWeights(0.25, 0.25, 0.25, 0.25)
        return LeadScoreWeights(
            contact=self.contact / total,
            company_size=self.company_size / total,
            activity=self.activity / total,
            web_recency=self.web_recency / total,
        )


@dataclass(frozen=True)
class LeadScoreBreakdown:
    """Breakdown of the different scoring components."""

    contact: float
    company_size: float
    activity: float
    web_recency: float
    score_business: float
    weights: LeadScoreWeights

    def as_dict(self) -> dict[str, float]:
        """Return a serializable representation of the score."""

        return {
            "contact": self.contact,
            "company_size": self.company_size,
            "activity": self.activity,
            "web_recency": self.web_recency,
            "score_business": self.score_business,
        }


DEFAULT_WEIGHTS = LeadScoreWeights()


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return any(_is_non_empty(item) for item in value)
    return True


def _extract_first(lead: Mapping[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in lead:
            value = lead[key]
            if _is_non_empty(value):
                return value
    return None


def _score_contact_completeness(lead: Mapping[str, Any]) -> float:
    weights = {
        "email": 0.35,
        "phone": 0.25,
        "website": 0.20,
        "address": 0.10,
        "linkedin": 0.10,
    }

    score = 0.0
    if _is_non_empty(_extract_first(lead, "best_email", "email", "email_pro")):
        score += weights["email"]
    if _is_non_empty(_extract_first(lead, "telephone_norm", "telephone", "phone")):
        score += weights["phone"]
    if _is_non_empty(_extract_first(lead, "site_web", "website", "domain_root")):
        score += weights["website"]
    if _is_non_empty(_extract_first(lead, "adresse_complete", "address", "adresse")):
        score += weights["address"]
    if _is_non_empty(_extract_first(lead, "linkedin_url", "linkedin")):
        score += weights["linkedin"]

    return min(score, 1.0)


_EMPLOYEE_RE = re.compile(r"(\d+)(?:\s*[–-]\s*(\d+))?")


def _parse_employee_count(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not math.isnan(raw):
        return float(raw)
    if not isinstance(raw, str):
        return None

    matches = _EMPLOYEE_RE.findall(raw.replace("\u202f", " "))
    if not matches:
        return None

    values = []
    for start, end in matches:
        try:
            start_val = float(start)
            end_val = float(end) if end else start_val
        except ValueError:
            continue
        values.append((start_val + end_val) / 2)

    if not values:
        return None
    return sum(values) / len(values)


_REVENUE_RE = re.compile(r"([\d.,]+)\s*([kKmMbB]?)")


def _parse_revenue(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not math.isnan(raw):
        return float(raw)
    if not isinstance(raw, str):
        return None

    cleaned = raw.replace("\u202f", " ").replace("€", "").lower()
    match = _REVENUE_RE.search(cleaned)
    if not match:
        return None

    number, suffix = match.groups()
    number = number.replace(" ", "").replace(",", ".")

    try:
        value = float(number)
    except ValueError:
        return None

    multiplier = 1.0
    if suffix == "k":
        multiplier = 1_000.0
    elif suffix == "m":
        multiplier = 1_000_000.0
    elif suffix == "b":
        multiplier = 1_000_000_000.0

    return value * multiplier


def _score_company_size(lead: Mapping[str, Any]) -> float:
    employee_value = _extract_first(
        lead,
        "employee_count",
        "effectif",
        "effectifs",
        "effectif_min",
        "effectif_max",
        "tranche_effectif",
    )
    employees = _parse_employee_count(employee_value)

    revenue_value = _extract_first(
        lead,
        "chiffre_affaires",
        "ca",
        "ca_brut",
        "revenue",
        "revenue_range",
        "turnover",
    )
    revenue = _parse_revenue(revenue_value)

    employee_score = 0.0
    if employees is not None:
        if employees >= 250:
            employee_score = 1.0
        elif employees >= 100:
            employee_score = 0.85
        elif employees >= 50:
            employee_score = 0.7
        elif employees >= 20:
            employee_score = 0.55
        elif employees >= 10:
            employee_score = 0.45
        elif employees >= 3:
            employee_score = 0.30
        elif employees > 0:
            employee_score = 0.15

    revenue_score = 0.0
    if revenue is not None:
        if revenue >= 50_000_000:
            revenue_score = 1.0
        elif revenue >= 10_000_000:
            revenue_score = 0.85
        elif revenue >= 2_000_000:
            revenue_score = 0.70
        elif revenue >= 500_000:
            revenue_score = 0.55
        elif revenue >= 100_000:
            revenue_score = 0.40
        elif revenue > 0:
            revenue_score = 0.20

    return max(employee_score, revenue_score)


def _score_activity(lead: Mapping[str, Any]) -> float:
    naf_value = _extract_first(
        lead,
        "naf",
        "naf_code",
        "ape",
        "ape_code",
        "code_naf",
        "code_ape",
    )

    if isinstance(naf_value, str):
        normalized = naf_value.strip().upper()
        if re.fullmatch(r"\d{4}[A-Z]", normalized):
            return 1.0
        if re.fullmatch(r"\d{3}[A-Z]?", normalized):
            return 0.75
        if re.fullmatch(r"\d{2}", normalized):
            return 0.55
        if len(normalized) >= 3:
            return 0.35

    activity_text = _extract_first(lead, "secteur_activite", "activity", "description")
    if isinstance(activity_text, str) and len(activity_text.strip()) >= 10:
        return 0.30

    return 0.0


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y",
)


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)) and not math.isnan(value):
        # Interpret numeric values as year when possible.
        year = int(value)
        if 1900 <= year <= date.today().year + 1:
            return date(year, 1, 1)
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return parsed.date()

    # Attempt ISO parsing for partially specified dates.
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        return None


def _score_web_recency(lead: Mapping[str, Any]) -> float:
    creation_date = _parse_date(_extract_first(lead, "date_creation", "date_de_creation"))
    if creation_date is None:
        return 0.0

    today = date.today()
    age_years = max(0.0, (today - creation_date).days / 365.25)

    if age_years <= 2:
        return 1.0
    if age_years <= 5:
        return 0.85
    if age_years <= 10:
        return 0.65
    if age_years <= 15:
        return 0.45
    if age_years <= 20:
        return 0.25
    return 0.10


def compute_lead_score(
    lead: Mapping[str, Any], *, weights: LeadScoreWeights = DEFAULT_WEIGHTS
) -> LeadScoreBreakdown:
    """Compute the business score for a single lead.

    Parameters
    ----------
    lead:
        Mapping containing lead information.
    weights:
        Optional custom weights for the different components.
    """

    normalized_weights = weights.normalized()
    contact_score = _score_contact_completeness(lead)
    company_size_score = _score_company_size(lead)
    activity_score = _score_activity(lead)
    web_recency_score = _score_web_recency(lead)

    overall = (
        contact_score * normalized_weights.contact
        + company_size_score * normalized_weights.company_size
        + activity_score * normalized_weights.activity
        + web_recency_score * normalized_weights.web_recency
    )

    return LeadScoreBreakdown(
        contact=contact_score,
        company_size=company_size_score,
        activity=activity_score,
        web_recency=web_recency_score,
        score_business=round(overall * 100, 2),
        weights=normalized_weights,
    )


def add_business_score(
    df: "pd.DataFrame", *, inplace: bool = False, weights: LeadScoreWeights = DEFAULT_WEIGHTS
) -> "pd.DataFrame":
    """Add a ``score_business`` column to a pandas DataFrame."""

    if pd is None:  # pragma: no cover - pandas is available in tests.
        raise ModuleNotFoundError("pandas is required to use add_business_score")

    target: "pd.DataFrame" = df if inplace else df.copy()
    target["score_business"] = target.apply(
        lambda row: compute_lead_score(row, weights=weights).score_business,
        axis=1,
    )
    return target


__all__ = [
    "LeadScoreBreakdown",
    "LeadScoreWeights",
    "DEFAULT_WEIGHTS",
    "compute_lead_score",
    "add_business_score",
]

