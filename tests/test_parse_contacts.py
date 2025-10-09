from pathlib import Path
from typing import Optional

import pandas as pd

from parse import parse_contacts


def _write_inputs(outdir: Path, pages_df: pd.DataFrame, serp_df: pd.DataFrame, dynamic_df: Optional[pd.DataFrame] = None):
    crawl_dir = outdir / "crawl"
    serp_dir = outdir / "serp"
    crawl_dir.mkdir()
    serp_dir.mkdir()
    pages_df.to_parquet(crawl_dir / "pages.parquet", index=False)
    serp_df.to_parquet(serp_dir / "serp_results.parquet", index=False)
    if dynamic_df is not None:
        headless_dir = outdir / "headless"
        headless_dir.mkdir()
        dynamic_df.to_parquet(headless_dir / "pages_dynamic.parquet", index=False)


def _run(outdir: Path) -> pd.DataFrame:
    cfg = {
        "quality": {"email_generic_list": ["contact@", "info@"]},
        "parse": {"prefer_mentions_legales": True},
    }
    ctx = {"outdir": str(outdir), "logger": None}
    result = parse_contacts.run(cfg, ctx)
    assert result["status"] == "OK"
    contacts_path = outdir / "contacts" / "contacts.parquet"
    assert contacts_path.exists()
    return pd.read_parquet(contacts_path)


def test_parse_contacts_prefers_siren_key_and_normalizes_email(tmp_path):
    outdir = tmp_path / "case_siren"
    outdir.mkdir()
    pages_df = pd.DataFrame(
        [
            {
                "domain": "Cabinet-Dupont.FR",
                "requested_url": "https://cabinet-dupont.fr/mentions-legales",
                "url": "https://cabinet-dupont.fr/mentions-legales",
                "status": 200,
                "content_type": "text/html",
                "content_text": (
                    "Cabinet Dupont SARL - Contact : Jean Dupont [at] cabinet-dupont.fr "
                    "Téléphone +33 (0)1 23 45 67 89 - SIRET 12345678900021"
                ),
                "content_html_trunc": "",
                "bytes": 512,
                "discovered_at": "2024-09-01T10:00:00+00:00",
                "siren_list": ["123456789"],
            }
        ]
    )
    serp_df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "denomination": "Cabinet Dupont",
                "top_url": "",
                "top_domain": "",
                "query": "Cabinet Dupont 75008 Paris",
                "rank": 1,
                "confidence": 92.5,
            }
        ]
    )
    _write_inputs(outdir, pages_df, serp_df)
    contacts_df = _run(outdir)
    assert len(contacts_df) == 1
    row = contacts_df.iloc[0]
    assert row["domain"] == "cabinet-dupont.fr"
    assert row["best_email"] == "jean.dupont@cabinet-dupont.fr"
    assert row["email_type"] == "nominative"
    assert "+33123456789" in row["phones"]
    assert "12345678900021" in row["sirets"]
    assert "123456789" in row["sirens_extracted"]


def test_parse_contacts_falls_back_to_domain_when_no_siren(tmp_path):
    outdir = tmp_path / "case_domain"
    outdir.mkdir()
    pages_df = pd.DataFrame(
        [
            {
                "domain": "atelier-lenoir.fr",
                "requested_url": "https://atelier-lenoir.fr/contact",
                "url": "https://atelier-lenoir.fr/contact",
                "status": 200,
                "content_type": "text/html",
                "content_text": (
                    "Écrivez-nous à contact@Atelier-Lenoir.fr ou marie.lenoir@atelier-lenoir.fr "
                    "Téléphone : 01 44 22 11 00"
                ),
                "content_html_trunc": "",
            },
            {
                "domain": "atelier-lenoir.fr",
                "requested_url": "https://atelier-lenoir.fr/assets/logo.png",
                "url": "https://atelier-lenoir.fr/assets/logo.png",
                "status": 200,
                "content_type": "image/png",
                "content_text": "",
                "content_html_trunc": "",
            },
        ]
    )
    serp_df = pd.DataFrame(
        [
            {
                "siren": None,
                "denomination": "Atelier Lenoir",
                "top_url": "https://atelier-lenoir.fr",
                "top_domain": "atelier-lenoir.fr",
                "query": "Atelier Lenoir",
                "rank": 2,
                "confidence": 76.4,
            }
        ]
    )
    _write_inputs(outdir, pages_df, serp_df)
    contacts_df = _run(outdir)
    assert len(contacts_df) == 1
    row = contacts_df.iloc[0]
    assert row["domain"] == "atelier-lenoir.fr"
    assert row["best_email"] == "marie.lenoir@atelier-lenoir.fr"
    assert row["email_type"] == "nominative"
    assert row["best_email"] in row["emails"]
    assert "contact@atelier-lenoir.fr" in row["emails"]
    assert "+33144221100" in row["phones"]


def test_parse_contacts_uses_dynamic_pages_for_tel_and_mail(tmp_path):
    outdir = tmp_path / "case_dynamic"
    outdir.mkdir()
    pages_df = pd.DataFrame(
        [
            {
                "domain": "exemple.fr",
                "requested_url": "https://exemple.fr",
                "url": "https://exemple.fr",
                "status": 200,
                "content_type": "text/html",
                "content_text": "",
                "content_html_trunc": "",
            }
        ]
    )
    dynamic_df = pd.DataFrame(
        [
            {
                "domain": "exemple.fr",
                "requested_url": "https://exemple.fr/contact",
                "url": "https://exemple.fr/contact",
                "status": 200,
                "content_type": "text/html",
                "content_html": (
                    "<a href='mailto:Contact@Exemple.fr'>Nous écrire</a>"
                    "<a href='tel:+33 (0)1 98 76 54 32'>Appeler</a>"
                ),
                "content_text": "",
            }
        ]
    )
    serp_df = pd.DataFrame(
        [
            {
                "siren": "",
                "denomination": "Exemple",
                "top_url": "https://exemple.fr",
                "top_domain": "exemple.fr",
                "query": "Exemple",
                "rank": 1,
                "confidence": 80.0,
            }
        ]
    )
    _write_inputs(outdir, pages_df, serp_df, dynamic_df=dynamic_df)
    contacts_df = _run(outdir)
    assert len(contacts_df) == 1
    row = contacts_df.iloc[0]
    assert row["best_email"] == "contact@exemple.fr"
    assert "+33198765432" in row["phones"]
    jsonl_path = outdir / "contacts" / "contacts.jsonl"
    assert jsonl_path.exists()
    assert jsonl_path.read_text().strip() != ""
