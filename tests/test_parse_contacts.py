from pathlib import Path

import pandas as pd

from parse import parse_contacts


def test_parse_contacts_extracts_data(tmp_path):
    outdir = tmp_path
    crawl_dir = outdir / "crawl"
    serp_dir = outdir / "serp"
    crawl_dir.mkdir()
    serp_dir.mkdir()

    pages_df = pd.DataFrame(
        [
            {
                "domain": "cabinet-dupont.fr",
                "requested_url": "https://cabinet-dupont.fr/mentions-legales",
                "url": "https://cabinet-dupont.fr/mentions-legales",
                "status": 200,
                "content_type": "text/html",
                "content_text": (
                    "Cabinet Dupont SARL - contact : Jean Dupont (at) cabinet-dupont.fr "
                    "Telephone 01 23 45 67 89 - SIRET 12345678900021"
                ),
                "content_html_trunc": "",
                "bytes": 512,
                "discovered_at": "2024-09-01T10:00:00+00:00",
                "siren_list": ["123456789"],
                "denominations": ["Cabinet Dupont"],
                "https": True,
            }
        ]
    )
    pages_df.to_parquet(crawl_dir / "pages.parquet", index=False)

    serp_df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "denomination": "Cabinet Dupont",
                "top_url": "https://cabinet-dupont.fr",
                "top_domain": "cabinet-dupont.fr",
                "query": "Cabinet Dupont 75008 Paris",
                "rank": 1,
                "confidence": 92.5,
            }
        ]
    )
    serp_df.to_parquet(serp_dir / "serp_results.parquet", index=False)

    cfg = {
        "quality": {"email_generic_list": ["contact@", "info@"]},
        "parse": {"prefer_mentions_legales": True},
    }
    ctx = {"outdir": str(outdir), "logger": None}

    result = parse_contacts.run(cfg, ctx)
    assert result["status"] == "OK"

    contacts_path = outdir / "contacts" / "contacts.parquet"
    assert contacts_path.exists()
    contacts_df = pd.read_parquet(contacts_path)
    assert len(contacts_df) == 1
    row = contacts_df.iloc[0]
    assert row["best_email"] == "jean.dupont@cabinet-dupont.fr"
    assert row["email_type"] == "nominative"
    assert "+33123456789" in row["phones"]
    assert "12345678900021" in row["sirets"]
    assert "123456789" in row["sirens_extracted"]

    jsonl_path = outdir / "contacts" / "contacts.jsonl"
    assert jsonl_path.exists()
    assert jsonl_path.read_text().strip() != ""
