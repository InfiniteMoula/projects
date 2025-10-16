import pandas as pd

from enrich.linkedin_clearbit_lite import process_linkedin_clearbit_lite


def test_process_naf_mapping_and_linkedin_slug():
    df = pd.DataFrame(
        [
            {
                "denomination": "Tech Care SAS",
                "naf": "62.01Z",
                "domain_root": "tech-care.fr",
            }
        ]
    )
    result, summary = process_linkedin_clearbit_lite(df, {})
    assert bool(result.loc[0, "has_domain"]) is True
    assert result.loc[0, "industry"] == "Information Technology & Services"
    assert result.loc[0, "linkedin_url"] == "https://www.linkedin.com/company/tech-care/"
    assert summary["with_domain"] == 1
    assert summary["linkedin_urls"] == 1


def test_process_keyword_fallback_on_name():
    df = pd.DataFrame(
        [
            {
                "denomination": "Cabinet d'Avocats Paris",
                "siteweb": "https://paris-avocats.fr",
            }
        ]
    )
    result, summary = process_linkedin_clearbit_lite(df, {})
    assert result.loc[0, "industry"] == "Legal Services"
    assert result.loc[0, "linkedin_url"].endswith("/paris-avocats/")
    assert summary["industry_filled"] == 1


def test_process_employee_range_from_insee_code():
    df = pd.DataFrame(
        [
            {
                "denomination": "Logistique Express",
                "domain_root": "logistique-express.fr",
                "trancheEffectifsUniteLegale": "21",
            }
        ]
    )
    result, summary = process_linkedin_clearbit_lite(df, {})
    assert result.loc[0, "employee_range"] == "51-200"
    assert summary["employee_range_filled"] == 1
