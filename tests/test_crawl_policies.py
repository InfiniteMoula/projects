import types

import pytest

from utils.robots import RobotsCache


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.headers = {}


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return self._response


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("https://example.org/private/data", False),
        ("https://example.org/public/info", True),
    ],
)
def test_robots_cache_respects_disallow(monkeypatch, path, allowed):
    robots_text = "User-agent: *\nDisallow: /private"
    fake_client = _FakeClient(_FakeResponse(robots_text))

    monkeypatch.setattr("utils.robots.httpx.Client", lambda *a, **k: fake_client)
    cache = RobotsCache(user_agent="TestBot/0.1", cache_ttl=0)
    assert cache.allowed(path, respect_robots=True) is allowed
