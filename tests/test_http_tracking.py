import json
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401  # ensure parquet support is available
import pytest

from utils.http import request_with_backoff
from nethttp import collect_serp
from enrich import google_maps_search


class _FakeHTTPError(Exception):
    pass


def test_request_with_backoff_tracks_bytes():
    called = {"bytes": []}

    class FakeResponse:
        def __init__(self):
            self.status_code = 200
            self._content = b"payload"

        @property
        def content(self):
            return self._content

    class FakeClient:
        def request(self, method, url, **kwargs):
            return FakeResponse()

    def tracker(size: int):
        called["bytes"].append(size)

    response = request_with_backoff(FakeClient(), "GET", "https://example.com", request_tracker=tracker)
    assert response.status_code == 200
    assert called["bytes"] == [len(b"payload")]


def _write_normalized(outdir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "denomination": "Example SARL",
                "ville": "Paris",
                "code_postal": "75001",
                "raison_sociale": "Example SARL",
                "numero_voie": "10",
                "type_voie": "rue",
                "libelle_voie": "du Test",
            }
        ]
    )
    df.to_parquet(outdir / "normalized.parquet", index=False)


class _FakeSerpResponse:
    def __init__(self):
        self.status_code = 200
        self.text = (
            '<ol id="b_results">'
            '  <li class="b_algo">'
            '    <a href="https://example.com">Example Company</a>'
            '    <p>Example snippet</p>'
            "  </li>"
            "</ol>"
        )
        self.content = self.text.encode("utf-8")


class _FakeSerpClient:
    request_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None, params=None):
        _FakeSerpClient.request_count += 1
        return _FakeSerpResponse()


def test_collect_serp_uses_sequential_state(tmp_path, monkeypatch):
    outdir = tmp_path / "run"
    outdir.mkdir()
    _write_normalized(outdir)

    monkeypatch.setattr(collect_serp.httpx, "Client", _FakeSerpClient)

    tracked_sizes = []

    def tracker(size: int):
        tracked_sizes.append(size)

    cfg = {"serp": {"max_results": 1}}
    ctx = {
        "outdir": str(outdir),
        "logger": None,
        "request_tracker": tracker,
    }

    _FakeSerpClient.request_count = 0
    result = collect_serp.run(cfg, ctx)
    assert result["status"] == "OK"
    assert _FakeSerpClient.request_count == 1
    state_file = outdir / "serp" / "serp_state.json"
    assert state_file.exists()
    state_payload = json.loads(state_file.read_text())
    assert state_payload["completed"]
    first_size_count = len(tracked_sizes)

    # second run should reuse state and avoid extra HTTP requests
    result_again = collect_serp.run(cfg, ctx)
    assert result_again["status"] == "OK"
    assert _FakeSerpClient.request_count == 1  # unchanged
    assert len(tracked_sizes) == first_size_count


class _FakeMapsResponse:
    def __init__(self):
        self.status_code = 200
        self._body = (
            "<html>"
            " Example Phone 01 23 45 67 89 "
            " contact@example.com "
            " <div>Business description</div>"
            "</html>"
        )

    @property
    def text(self):
        return self._body

    @property
    def content(self):
        return self._body.encode("utf-8")


class _FakeMapsClient:
    request_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, timeout=None, follow_redirects=True):
        _FakeMapsClient.request_count += 1
        return _FakeMapsResponse()


def _write_database(outdir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "index": 0,
                "adresse": "10 rue du Test Paris 75001",
                "company_name": "Example SARL",
                "siren": "123456789",
                "siret": "12345678900011",
            }
        ]
    )
    df.to_csv(outdir / "database.csv", index=False)


def test_google_maps_search_resumes_from_state(tmp_path, monkeypatch):
    outdir = tmp_path / "maps"
    outdir.mkdir()
    _write_normalized(outdir)
    _write_database(outdir)

    monkeypatch.setattr(google_maps_search.httpx, "Client", _FakeMapsClient)

    tracked_sizes = []

    def tracker(size: int):
        tracked_sizes.append(size)

    ctx = {
        "outdir": str(outdir),
        "logger": None,
        "request_tracker": tracker,
    }

    cfg = {"maps": {}}

    _FakeMapsClient.request_count = 0
    result = google_maps_search.run(cfg, ctx)
    assert result["status"] == "OK"
    assert _FakeMapsClient.request_count == 1
    state_file = outdir / "google_maps_state.json"
    assert state_file.exists()
    state_payload = json.loads(state_file.read_text())
    assert state_payload["completed"]
    initial_sizes = len(tracked_sizes)

    second = google_maps_search.run(cfg, ctx)
    assert second["status"] == "OK"
    assert _FakeMapsClient.request_count == 1
    assert len(tracked_sizes) == initial_sizes
