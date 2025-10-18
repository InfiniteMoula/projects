from __future__ import annotations

import pytest

from api.pappers_client import PappersAPIError, fetch_pappers_company


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.responses:
            raise AssertionError("No more responses configured for DummyClient")
        return DummyResponse(self.responses.pop(0))

    def close(self):
        return None


def test_fetch_company_with_siren_uses_direct_endpoint():
    entreprise_payload = {
        "denomination": "OpenAI France",
        "siren": "123456789",
        "siege": {
            "adresse_ligne_1": "1 Rue de l'Exemple",
            "code_postal": "75000",
            "ville": "Paris",
        },
        "capital": 1000,
        "effectif": "10-19",
        "dirigeants": [
            {"nom": "Dupont", "prenom": "Jean", "fonction": "Président"}
        ],
        "statut_rcs": "actif",
        "code_naf": "62.01Z",
        "site_web": "https://openai.fr",
    }

    client = DummyClient([entreprise_payload])

    data = fetch_pappers_company("123456789", api_token="token", client=client)

    assert data == {
        "nom": "OpenAI France",
        "siren": "123456789",
        "adresse": "1 Rue de l'Exemple, 75000, Paris",
        "capital": 1000,
        "effectif": "10-19",
        "dirigeants": [
            {"nom": "Dupont", "prenom": "Jean", "fonction": "Président"}
        ],
        "statut": "actif",
        "code_naf": "62.01Z",
        "site_web": "https://openai.fr",
    }

    assert len(client.calls) == 1
    assert client.calls[0]["url"].endswith("entreprise")


def test_fetch_company_with_name_uses_search_then_entreprise():
    search_payload = {"resultats": [{"siren": "987654321"}]}
    entreprise_payload = {
        "nom_entreprise": "ACME",
        "siren": "987654321",
        "siege": {
            "adresse_ligne_1": "2 Avenue Exemple",
            "adresse_ligne_2": "Bâtiment B",
            "code_postal": "69000",
            "ville": "Lyon",
        },
        "capital_actuel": 5000,
        "tranche_effectif_salarie": "20-49",
        "representants": [
            {"nom": "Martin", "prenom": "Claire", "qualite": "Gérant"}
        ],
        "est_entreprise_active": False,
        "code_naf_entreprise": "47.91A",
        "site_internet": "https://acme.fr",
    }

    client = DummyClient([search_payload, entreprise_payload])

    data = fetch_pappers_company("ACME", api_token="token", client=client)

    assert data == {
        "nom": "ACME",
        "siren": "987654321",
        "adresse": "2 Avenue Exemple, Bâtiment B, 69000, Lyon",
        "capital": 5000,
        "effectif": "20-49",
        "dirigeants": [
            {"nom": "Martin", "prenom": "Claire", "fonction": "Gérant"}
        ],
        "statut": "inactif",
        "code_naf": "47.91A",
        "site_web": "https://acme.fr",
    }

    assert len(client.calls) == 2
    assert client.calls[0]["url"].endswith("recherche")
    assert client.calls[1]["url"].endswith("entreprise")


def test_missing_results_raise_value_error():
    client = DummyClient([{"resultats": []}])

    with pytest.raises(ValueError):
        fetch_pappers_company("Unknown", api_token="token", client=client)


def test_empty_payload_raises_api_error():
    client = DummyClient([{"resultats": [{"siren": "987654321"}]}, {}])

    with pytest.raises(PappersAPIError):
        fetch_pappers_company("ACME", api_token="token", client=client)

