import json
from pathlib import Path

import pandas as pd

from scraper import maps_scraper

SAMPLE_HTML = """
<html>
  <head>
    <script type="application/ld+json">
    {
      "@context": "http://schema.org",
      "@type": "LocalBusiness",
      "name": "Cabinet Expert Comptable",
      "address": {
        "streetAddress": "10 Rue Exemple",
        "postalCode": "75001",
        "addressLocality": "Paris",
        "addressCountry": "FR"
      },
      "telephone": "+33 1 23 45 67 89",
      "url": "https://cabinet-expert.fr",
      "aggregateRating": {
        "ratingValue": "4.6",
        "reviewCount": "128"
      },
      "@id": "https://maps.google.com/?cid=123456"
    }
    </script>
  </head>
  <body></body>
</html>
"""


def _write_input(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "denomination": "Cabinet Expert Comptable",
                "city": "Paris",
                "postal_code": "75001",
            }
        ]
    )
    input_path = tmp_path / "normalized.parquet"
    frame.to_parquet(input_path, index=False)
    return input_path


def test_maps_scraper_extracts_fields(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)

    def fake_retrieve(self, query, siren):  # pragma: no cover - simple monkeypatch helper
        return SAMPLE_HTML, "https://maps.google.com/?cid=123456"

    monkeypatch.setattr(maps_scraper.MapsScraper, "_retrieve_html", fake_retrieve)

    result = maps_scraper.run(
        {
            "delay_range": (0.0, 0.0),
            "input_path": str(input_path),
            "max_retries": 0,
            "per_host_rps": 10.0,
        },
        {"outdir_path": tmp_path},
    )

    assert result["status"] == "OK"
    assert result["rows"] == 1

    parquet_path = tmp_path / "maps" / "maps_results.parquet"
    jsonl_path = tmp_path / "maps" / "maps_results.jsonl"

    assert parquet_path.exists()
    assert jsonl_path.exists()

    enriched = pd.read_parquet(parquet_path)
    assert enriched.loc[0, "phone"] == "+33 1 23 45 67 89"
    assert enriched.loc[0, "website"] == "https://cabinet-expert.fr"
    assert enriched.loc[0, "address_complete"].startswith("10 Rue Exemple")
    assert enriched.loc[0, "reviews_count"] == 128
    assert enriched.loc[0, "rating_avg"] == 4.6
    assert enriched.loc[0, "google_maps_url"].startswith("https://maps.google.com/")
    assert enriched.loc[0, "maps_confidence_score"] >= 80

    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["phone"] == "+33 1 23 45 67 89"
    assert payload["maps_confidence_score"] >= 80


def test_maps_scraper_handles_missing_data(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)

    def fake_retrieve(self, query, siren):  # pragma: no cover - simple monkeypatch helper
        return None, None

    monkeypatch.setattr(maps_scraper.MapsScraper, "_retrieve_html", fake_retrieve)

    result = maps_scraper.run(
        {
            "delay_range": (0.0, 0.0),
            "input_path": str(input_path),
            "max_retries": 0,
            "per_host_rps": 10.0,
        },
        {"outdir_path": tmp_path},
    )

    assert result["status"] == "OK"
    assert result["rows"] == 0

    parquet_path = tmp_path / "maps" / "maps_results.parquet"
    enriched = pd.read_parquet(parquet_path)
    assert enriched.empty
