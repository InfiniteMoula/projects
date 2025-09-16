import pytest
from nethttp import collect_http_static
from utils import budget_middleware


class _FakeResponse:
    def __init__(self, url: str):
        self.content = f"<html>{url}</html>".encode()
        self.text = f"<html>{url}</html>"
        self.status_code = 200


class _FakeClient:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        self.calls.append(url)
        return _FakeResponse(url)


def test_collect_http_static_stops_on_global_budget(tmp_path, monkeypatch):
    tracker = budget_middleware.BudgetTracker(max_http_requests=1)
    ctx = {
        "outdir": str(tmp_path),
        "budget_tracker": tracker,
        "request_tracker": tracker.track_http_request,
        "logger": None,
    }
    cfg = {
        "http": {
            "seeds": ["https://example.com/one", "https://example.com/two"],
            "per_domain_rps": 0,
        }
    }

    fake_client = _FakeClient()
    monkeypatch.setattr(
        collect_http_static.httpx,
        "Client",
        lambda *args, **kwargs: fake_client,
    )

    with pytest.raises(budget_middleware.BudgetExceededError):
        collect_http_static.run(cfg, ctx)

    assert fake_client.calls == ["https://example.com/one"]
    saved_files = list((tmp_path / "http").glob("*.html"))
    assert len(saved_files) == 1
    assert tracker.http_requests == 1

