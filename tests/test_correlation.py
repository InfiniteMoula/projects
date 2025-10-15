import json
from pathlib import Path

import pandas as pd

from correlation import correlation


def _run(df: pd.DataFrame, tmp_path: Path, **cfg):
    cfg.setdefault("report_path", tmp_path / "report.json")
    cfg.setdefault("country_field", "country_code")
    cfg.setdefault("default_country", "FR")
    return correlation.run(df, cfg)


def test_email_domain_match_bonus(tmp_path):
    df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "site_web": "https://www.example.com",
                "email": "jean@example.com",
                "country_code": "FR",
            }
        ]
    )
    result = _run(df, tmp_path)
    assert result.loc[0, "score_coherence"] == 80.0
    assert "email_domain_match" in result.loc[0, "coherence_flags"]


def test_email_domain_mismatch_penalises(tmp_path):
    df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "site_web": "https://www.example.com",
                "email": "jean@other.org",
                "country_code": "FR",
            }
        ]
    )
    result = _run(df, tmp_path)
    assert result.loc[0, "score_coherence"] == 30.0
    assert "email_domain_mismatch" in result.loc[0, "coherence_flags"]


def test_phone_country_mismatch_penalises(tmp_path):
    df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "telephone_norm": "+441234567890",
                "country_code": "FR",
            }
        ]
    )
    result = _run(df, tmp_path)
    assert result.loc[0, "score_coherence"] == 30.0
    assert "phone_country_mismatch" in result.loc[0, "coherence_flags"]


def test_linkedin_mismatch_penalises(tmp_path):
    df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "site_web": "https://www.example.com",
                "linkedin_url": "https://www.linkedin.com/company/notexample/",
                "country_code": "FR",
            }
        ]
    )
    result = _run(df, tmp_path)
    assert result.loc[0, "score_coherence"] == 35.0
    assert "linkedin_site_mismatch" in result.loc[0, "coherence_flags"]


def test_generic_email_penalty(tmp_path):
    df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "email": "contact@gmail.com",
                "country_code": "FR",
            }
        ]
    )
    result = _run(df, tmp_path)
    assert result.loc[0, "score_coherence"] == 40.0
    assert any(flag.startswith("generic_email") for flag in result.loc[0, "coherence_flags"])


def test_run_full_flow_generates_report(tmp_path):
    report_path = tmp_path / "correlation_report.json"
    df = pd.DataFrame(
        [
            {
                "siren": "111",
                "siret": "11100011100011",
                "site_web": "https://www.alpha.fr",
                "email": "contact@alpha.fr",
                "telephone_norm": "+33123456789",
                "linkedin_url": "https://www.linkedin.com/company/alpha/",
                "country_code": "FR",
            },
            {
                "siren": "222",
                "siret": "22200022200022",
                "site_web": "https://www.beta.fr",
                "email": "support@gmail.com",
                "telephone_norm": "+441234567890",
                "linkedin_url": "https://www.linkedin.com/company/notbeta/",
                "country_code": "FR",
            },
            {
                "siren": "333",
                "siret": "33300033300033",
                "email": "team@other.com",
                "country_code": "FR",
            },
        ]
    )
    result = correlation.run(
        df,
        {
            "report_path": report_path,
            "country_field": "country_code",
            "default_country": "FR",
            "top_incoherences": 2,
        },
    )
    assert "score_coherence" in result.columns
    assert "coherence_flags" in result.columns
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_rows"] == 3
    assert payload["incoherent_rows"] >= 1
    assert "score_distribution" in payload
    assert len(payload["top_incoherences"]) <= 2
