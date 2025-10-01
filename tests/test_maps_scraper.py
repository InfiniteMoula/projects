import json
from pathlib import Path

import pandas as pd
import pytest

from scraper import maps_scraper

SAMPLE_INPUT = {
    "siren": "123456789",
    "denomination": "Cabinet Expert Comptable",
    "city": "Paris",
    "postal_code": "75001",
}

HTML_JSON_LD_GRAPH = """
<html>
  <head>
    <script type="application/ld+json">
    {
      "@context": "http://schema.org",
      "@graph": [
        {
          "@type": "LocalBusiness",
          "name": "Cabinet Expert Comptable",
          "address": {
            "streetAddress": "10 Rue Exemple",
            "postalCode": "75001",
            "addressLocality": "Paris",
            "addressCountry": "FR"
          },
          "telephone": "+33 1 23 45 67 89",
          "url": "cabinet-expert.fr",
          "aggregateRating": {
            "ratingValue": "4.6",
            "reviewCount": "128"
          },
          "@id": "https://maps.google.com/?cid=123456"
        }
      ]
    }
    </script>
  </head>
  <body></body>
</html>
"""

HTML_MICRODATA = """
<html>
  <body>
    <div itemscope itemtype="http://schema.org/LocalBusiness">
      <span itemprop="name">Cabinet Expert Comptable</span>
      <div itemprop="address" itemscope itemtype="http://schema.org/PostalAddress">
        <span itemprop="streetAddress">10 Rue Exemple</span>
        <span itemprop="postalCode">75001</span>
        <span itemprop="addressLocality">Paris</span>
        <span itemprop="addressCountry">FR</span>
      </div>
      <span itemprop="telephone">01 23 45 67 89</span>
      <a itemprop="url" href="https://cabinet-expert.fr">Site</a>
      <span itemprop="ratingValue">4.8</span>
      <span itemprop="reviewCount">45</span>
    </div>
  </body>
</html>
"""

HTML_AF_CALLBACK = """
<html>
  <head>
    <script>
      AF_initDataCallback({
        "key": "ds:1",
        "data": [[[{"name": "Cabinet Expert Comptable", "address": "10 Rue Exemple 75001 Paris FR", "phone": "+33 (0)1 23 45 67 89", "website": "https://cabinet-expert.fr", "reviews": 90, "rating": 4.4, "mapsUrl": "https://maps.google.com/?cid=999"}]]],
        "sideChannel": {}
      });
    </script>
  </head>
</html>
"""

HTML_LOCAL_PACK = """
<html>
  <head>
    <script id="local-pack-data" type="application/json">
    {
      "results": [
        {
          "name": "Cabinet Expert Comptable",
          "address": "10 Rue Exemple 75001 Paris FR",
          "phone": "+33 1 23 45 67 89",
          "website": "https://cabinet-expert.fr",
          "rating": 4.2,
          "reviews": 67,
          "mapsUrl": "https://maps.google.com/?cid=555"
        }
      ]
    }
    </script>
  </head>
</html>
"""


@pytest.fixture
def input_parquet(tmp_path: Path) -> Path:
    frame = pd.DataFrame([SAMPLE_INPUT])
    target = tmp_path / "normalized.parquet"
    frame.to_parquet(target, index=False)
    return target


def _run_with_html(tmp_path: Path, input_path: Path, monkeypatch: pytest.MonkeyPatch, localpack: str | None, maps: str | None, af: str | None = None) -> dict:
    html_by_mode = {
        "localpack": localpack,
        "maps": maps,
    }

    def fake_get_html(self, query: str, url: str, siren: str, mode: str):  # type: ignore[override]
        html = html_by_mode.get(mode)
        if html is None:
            return None, None
        return html, f"https://maps.google.com/{mode}/{siren}"

    def fake_playwright(self, query: str, siren: str):  # type: ignore[override]
        if af:
            return af, "https://maps.google.com/playwright"
        return None, None

    monkeypatch.setattr(maps_scraper.MapsScraper, "_get_html", fake_get_html)
    monkeypatch.setattr(maps_scraper.MapsScraper, "_fetch_with_playwright", fake_playwright)

    result = maps_scraper.run(
        {
            "delay_range": (0.0, 0.0),
            "timeout": 5,
            "batch_size": 16,
            "input_path": str(input_path),
            "per_host_rps": 10.0,
            "use_playwright": bool(af),
        },
        {"outdir_path": tmp_path},
    )
    return result


@pytest.mark.parametrize(
    "localpack_html,maps_html,af_html",
    [
        (HTML_LOCAL_PACK, None, None),
        (None, HTML_JSON_LD_GRAPH, None),
        (None, HTML_AF_CALLBACK, None),
        (None, HTML_MICRODATA, None),
    ],
)
def test_maps_scraper_extracts_fields(tmp_path: Path, input_parquet: Path, monkeypatch: pytest.MonkeyPatch, localpack_html: str | None, maps_html: str | None, af_html: str | None) -> None:
    result = _run_with_html(tmp_path, input_parquet, monkeypatch, localpack_html, maps_html, af_html)

    assert result["status"] == "OK"
    assert result["rows"] == 1

    parquet_path = Path(result["file"])
    jsonl_path = Path(result["jsonl"])
    metrics_path = Path(result["metrics_file"])

    assert parquet_path.exists()
    assert jsonl_path.exists()
    assert metrics_path.exists()

    enriched = pd.read_parquet(parquet_path)
    row = enriched.iloc[0]
    assert row["phone"].startswith("+33")
    assert row["website"].startswith("https://")
    assert row["address_complete"].lower().startswith("10 rue")
    assert row["rating_avg"] is None or row["rating_avg"] >= 0
    assert row["reviews_count"] is None or row["reviews_count"] >= 0
    assert row["maps_confidence_score"] >= 80

    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["maps_confidence_score"] >= 80

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "requests_made" in metrics


def test_maps_scraper_handles_missing_data(tmp_path: Path, input_parquet: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_with_html(tmp_path, input_parquet, monkeypatch, None, None)
    assert result["status"] == "OK"
    assert result["rows"] == 0

    parquet_path = Path(result["file"])
    enriched = pd.read_parquet(parquet_path)
    assert enriched.empty

