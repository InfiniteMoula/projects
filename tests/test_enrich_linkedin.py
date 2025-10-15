import pytest

from serp.providers import Result

from enrich.enrich_linkedin import (
    _normalize_text,
    _prepare_search_text,
    _score_result,
    _select_linkedin_result,
)


def test_prepare_search_text_removes_legal_form_and_accents():
    normalized = _prepare_search_text("Entreprise Élite SAS")
    assert normalized == "entreprise elite"


def test_score_result_exact_vs_partial():
    norm_name = _normalize_text("Entreprise Elite")
    exact = _score_result("Entreprise Elite | LinkedIn", norm_name)
    partial = _score_result("Entreprise Elite Paris - LinkedIn", norm_name)
    unrelated = _score_result("Autre Société", norm_name)
    assert exact == 1.0
    assert partial == pytest.approx(0.8, rel=1e-6)
    assert unrelated == 0.0


def test_select_linkedin_result_filters_user_profiles():
    norm_name = _normalize_text("Entreprise Elite")
    results = [
        Result(
            url="https://www.linkedin.com/in/john-doe/",
            domain="linkedin.com",
            title="John Doe",
            snippet="Personal profile",
            rank=1,
        ),
        Result(
            url="https://www.linkedin.com/company/entreprise-elite/",
            domain="linkedin.com",
            title="Entreprise Elite | LinkedIn",
            snippet="Entreprise Elite official page",
            rank=2,
        ),
    ]
    selected = _select_linkedin_result(results, norm_name)
    assert selected is not None
    url, score = selected
    assert url.endswith("/entreprise-elite/")
    assert score == 1.0
