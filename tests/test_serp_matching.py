import pytest

from nethttp.collect_serp import CompanyQuery, SerpSelection, _select_best, _normalize_name


def test_select_best_result_prefers_matching_domain():
    company = CompanyQuery(
        siren="123456789",
        denomination="Cabinet Dupont",
        city="Paris",
        postal_code="75008",
        address="10 Rue de la Paix 75008 Paris",
    )
    candidates = [
        {
            "url": "https://annuaire-professionnel.fr/dupont",
            "domain": "annuaire-professionnel.fr",
            "title": "Annuaire - Dupont Cabinet",
            "snippet": "Trouvez le cabinet Dupont sur notre annuaire",
            "rank": 1,
        },
        {
            "url": "https://cabinet-dupont.fr/contact",
            "domain": "cabinet-dupont.fr",
            "title": "Cabinet Dupont - Experts Comptables",
            "snippet": "Le cabinet Dupont accompagne les entreprises parisiennes",
            "rank": 2,
        },
    ]

    choice = _select_best(candidates, company)
    assert isinstance(choice, SerpSelection)
    assert choice.domain == "cabinet-dupont.fr"
    assert choice.rank == 2
    assert choice.confidence >= 70


def test_normalize_name_removes_legal_suffix():
    cleaned = _normalize_name("CABINET DUPONT SARL")
    assert cleaned == "CABINET DUPONT"
