from pathlib import Path

import pandas as pd

from parse import parse_contacts_ai
from quality import clean_contacts


def test_parse_contacts_ai_skips_without_tesseract(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_contacts_ai, "pytesseract", None)
    result = parse_contacts_ai.run({}, {"outdir": str(tmp_path), "logger": None})
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "PYTESSERACT_MISSING"


def test_clean_contacts_merges_ai_results(tmp_path):
    outdir = tmp_path
    contacts_dir = outdir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "domain": "example.com",
                "emails": [],
                "phones": [],
                "best_page": None,
                "best_status": None,
            }
        ]
    )
    base_df.to_parquet(contacts_dir / "contacts.parquet", index=False)

    ai_df = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "domain": "example.com",
                "emails": ["info@example.com"],
                "phones": ["+33123456789"],
            }
        ]
    )
    ai_df.to_parquet(contacts_dir / "contacts_ai.parquet", index=False)

    cfg = {"experimental": {"ai_ocr": True}}
    ctx = {"outdir": str(outdir), "logger": None}

    result = clean_contacts.run(cfg, ctx)
    assert result["status"] == "OK"

    clean_path = contacts_dir / "contacts_clean.parquet"
    df = pd.read_parquet(clean_path)
    row = df.iloc[0]
    assert "info@example.com" in row["emails"]
    assert "+33123456789" in row["phones"]
