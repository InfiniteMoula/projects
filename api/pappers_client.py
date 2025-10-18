"""Client helpers for fetching company data from the public Pappers API."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx


PAPPERS_SEARCH_URL = "https://api.pappers.fr/v2/recherche"
PAPPERS_ENTREPRISE_URL = "https://api.pappers.fr/v2/entreprise"


class PappersAPIError(RuntimeError):
    """Raised when the Pappers API returns an unexpected response."""


def fetch_pappers_company(
    query: str,
    *,
    api_token: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Return structured information about a company using the Pappers API.

    Parameters
    ----------
    query:
        Either the company SIREN (9 digits) or a free-text name that will be
        resolved using the search endpoint.
    api_token:
        The Pappers API token. If not provided, the ``PAPPERS_API_TOKEN``
        environment variable will be used. The free tier of the Pappers API
        requires an API token.
    client:
        Optional :class:`httpx.Client` instance. Mainly useful for testing.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``adresse``, ``capital``, ``effectif``,
        ``dirigeants``, ``statut``, ``code_naf`` and ``site_web``.

    Raises
    ------
    ValueError
        If the API token is missing or if the company cannot be found.
    PappersAPIError
        If the API response is malformed.
    httpx.HTTPError
        If the HTTP request fails.
    """

    token = api_token or os.getenv("PAPPERS_API_TOKEN")
    if not token:
        raise ValueError(
            "A Pappers API token is required. Set the 'PAPPERS_API_TOKEN' "
            "environment variable or pass api_token explicitly."
        )

    cleaned_query = _normalise_query(query)
    http = client or httpx.Client(timeout=30.0)
    created_client = client is None

    try:
        if _looks_like_siren(cleaned_query):
            siren = cleaned_query
        else:
            siren = _search_company_siren(http, token, cleaned_query)

        entreprise_data = _fetch_company_data(http, token, siren)
        if not entreprise_data:
            raise PappersAPIError("Empty payload returned by the entreprise endpoint")

        return _normalise_company_payload(entreprise_data)
    finally:
        if created_client:
            http.close()


def _normalise_query(query: str) -> str:
    return query.strip()


def _looks_like_siren(query: str) -> bool:
    return bool(re.fullmatch(r"\d{9}", query))


def _search_company_siren(http: httpx.Client, token: str, query: str) -> str:
    response = http.get(
        PAPPERS_SEARCH_URL,
        params={
            "api_token": token,
            "q": query,
            "par_page": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("resultats", [])
    if not results:
        raise ValueError(f"No company found for query '{query}'")

    siren = results[0].get("siren")
    if not siren:
        raise PappersAPIError("Missing SIREN in search result")

    return siren


def _fetch_company_data(http: httpx.Client, token: str, siren: str) -> Dict[str, Any]:
    response = http.get(
        PAPPERS_ENTREPRISE_URL,
        params={"api_token": token, "siren": siren},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _normalise_company_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    siege = payload.get("siege", {}) or {}

    return {
        "nom": _first_value(payload, ["denomination", "nom_entreprise", "nom"]),
        "siren": _first_value(payload, ["siren"]),
        "adresse": _format_address(siege),
        "capital": _first_value(payload, ["capital", "capital_actuel"]),
        "effectif": _extract_effectif(payload),
        "dirigeants": _extract_dirigeants(payload),
        "statut": _extract_statut(payload),
        "code_naf": _first_value(payload, ["code_naf", "code_naf_entreprise"]),
        "site_web": _first_value(payload, ["site_web", "site_internet", "url_site_web"]),
    }


def _first_value(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_address(siege: Dict[str, Any]) -> Optional[str]:
    if not siege:
        return None

    components: List[str] = []
    for key in (
        "adresse_ligne_1",
        "adresse_ligne_2",
        "adresse_ligne_3",
        "code_postal",
        "ville",
        "pays",
    ):
        value = siege.get(key)
        if value:
            components.append(str(value))

    return ", ".join(components) if components else None


def _extract_effectif(payload: Dict[str, Any]) -> Optional[str]:
    effectif = _first_value(
        payload,
        [
            "effectif",
            "tranche_effectif_salarie",
            "tranche_effectif_insee",
        ],
    )

    if effectif:
        return str(effectif)

    min_effectif = payload.get("effectif_min")
    max_effectif = payload.get("effectif_max")
    if min_effectif or max_effectif:
        return f"{min_effectif or '?'}-{max_effectif or '?'}"

    return None


def _extract_dirigeants(payload: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    dirigeants: List[Dict[str, Optional[str]]] = []
    for entry in payload.get("dirigeants", []) or []:
        dirigeants.append(
            {
                "nom": entry.get("nom"),
                "prenom": entry.get("prenom"),
                "fonction": entry.get("fonction") or entry.get("qualite"),
            }
        )

    if dirigeants:
        return dirigeants

    for entry in payload.get("representants", []) or []:
        dirigeants.append(
            {
                "nom": entry.get("nom"),
                "prenom": entry.get("prenom"),
                "fonction": entry.get("qualite"),
            }
        )

    return dirigeants


def _extract_statut(payload: Dict[str, Any]) -> Optional[str]:
    status = _first_value(payload, ["statut_rcs", "etat_administratif", "etat_insee"])
    if status:
        return str(status)

    active = payload.get("est_entreprise_active")
    if active is None:
        active = payload.get("siege", {}).get("est_siege_actif")

    if active is True:
        return "actif"
    if active is False:
        return "inactif"

    return None

