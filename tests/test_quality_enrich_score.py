from datetime import datetime, timedelta, timezone

import pandas as pd

from quality import enrich_score


def test_enrich_score_computes_weights(tmp_path):
    contacts_dir = tmp_path / "contacts"
    contacts_dir.mkdir()
    now_iso = datetime.now(timezone.utc).isoformat()
    old_iso = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()

    contacts_df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "denomination": "Cabinet Dupont",
                "domain": "cabinet-dupont.fr",
                "top_url": "https://cabinet-dupont.fr",
                "best_page": "https://cabinet-dupont.fr/mentions-legales",
                "best_status": 200,
                "best_email": "jean.dupont@cabinet-dupont.fr",
                "email_type": "nominative",
                "emails": ["jean.dupont@cabinet-dupont.fr"],
                "phones": ["+33123456789"],
                "social_links": {"linkedin": ["https://www.linkedin.com/company/cabinet-dupont"]},
                "rcs": ["RCS Paris 123456789"],
                "legal_managers": ["Jean Dupont"],
                "addresses": ["10 Rue de la Paix 75008 Paris"],
                "sirets": ["12345678900021"],
                "sirens_extracted": ["123456789"],
                "confidence": 92.0,
                "query": "Cabinet Dupont Paris",
                "rank": 1,
                "discovered_at": now_iso,
            },
            {
                "siren": "987654321",
                "denomination": "Cabinet Martin",
                "domain": "cabinet-martin.fr",
                "top_url": "http://cabinet-martin.fr",
                "best_page": "http://cabinet-martin.fr",
                "best_status": 404,
                "best_email": None,
                "email_type": None,
                "emails": [],
                "phones": [],
                "social_links": {},
                "rcs": [],
                "legal_managers": [],
                "addresses": [],
                "sirets": [],
                "sirens_extracted": [],
                "confidence": 10.0,
                "query": "Cabinet Martin Paris",
                "rank": 3,
                "discovered_at": old_iso,
            },
        ]
    )
    contacts_path = contacts_dir / "contacts.parquet"
    contacts_df.to_parquet(contacts_path, index=False)

    ctx = {"outdir": str(tmp_path), "logger": None}
    result = enrich_score.run({}, ctx)
    assert result["status"] == "OK"

    updated = pd.read_parquet(contacts_path)
    assert {"contactability_score", "completeness_score", "unicity_score", "freshness_score", "enrich_score"} <= set(updated.columns)

    high = updated.loc[updated["siren"] == "123456789"].iloc[0]
    low = updated.loc[updated["siren"] == "987654321"].iloc[0]

    assert high["enrich_score"] > 90
    assert high["contactability_score"] > 45
    assert high["freshness_score"] == 10

    assert low["enrich_score"] == 0
    assert low["contactability_score"] == 0
    assert low["freshness_score"] == 0

    report_path = tmp_path / "reports" / "enrichment_report.html"
    assert report_path.exists()
    assert "Rapport" in report_path.read_text()
