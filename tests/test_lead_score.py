from datetime import date

import pandas as pd

from ml.lead_score import add_business_score, compute_lead_score


def test_compute_lead_score_high_quality_lead():
    lead = {
        "best_email": "sales@example.com",
        "telephone_norm": "+33123456789",
        "site_web": "https://example.com",
        "adresse_complete": "10 Rue de la Paix, 75002 Paris",
        "linkedin_url": "https://www.linkedin.com/company/example",
        "effectif": "120",
        "chiffre_affaires": "12M€",
        "naf_code": "6202A",
        "date_creation": date.today().isoformat(),
    }

    score = compute_lead_score(lead)
    assert score.contact >= 0.9
    assert score.company_size >= 0.8
    assert score.activity == 1.0
    assert score.web_recency == 1.0
    assert score.score_business > 85


def test_compute_lead_score_low_information_lead():
    lead = {
        "activity": "",
        "date_creation": "1998-01-01",
    }

    score = compute_lead_score(lead)
    assert score.contact == 0.0
    assert score.company_size == 0.0
    assert score.activity == 0.0
    assert score.web_recency <= 0.25
    assert score.score_business < 30


def test_add_business_score_dataframe():
    df = pd.DataFrame(
        [
            {
                "email": "info@startup.com",
                "telephone": "0123456789",
                "site_web": "https://startup.com",
                "effectif": "10-19",
                "naf": "7022Z",
                "date_creation": date.today().year,
            },
            {
                "naf": "70",
                "date_creation": "2000",
            },
        ]
    )

    scored = add_business_score(df)
    assert "score_business" in scored.columns
    assert scored.loc[0, "score_business"] > scored.loc[1, "score_business"]
