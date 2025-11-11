import httpx
import pandas as pd

from serp.providers import BingProvider
from enrich import site_web_search


def test_prepare_search_text_strips_legal_form():
    assert site_web_search._prepare_search_text("ABC SAS") == "abc"
    assert site_web_search._prepare_search_text("Entreprise Élite SARL") == "entreprise elite"


def test_bing_provider_parses_results(monkeypatch):
    html = """
    <html>
      <body>
        <ul>
          <li class="b_algo">
            <h2><a href="https://example.com/index.html?utm=track">Example Title</a></h2>
            <div class="b_caption"><p>Snippet about Example company.</p></div>
          </li>
        </ul>
      </body>
    </html>
    """

    class StubHttpClient:
        def get(self, url: str):
            request = httpx.Request("GET", url)
            return httpx.Response(status_code=200, text=html, request=request)

    provider = BingProvider({"max_results": 5}, StubHttpClient())
    results = provider.search("test query")

    assert len(results) == 1
    result = results[0]
    assert result.url == "https://example.com/index.html"
    assert result.title == "Example Title"
    assert result.domain == "example.com"
    assert result.rank == 1


def test_site_web_search_uses_human_readable_name_column(monkeypatch):
    df = pd.DataFrame({"Nom entreprise": ["Alpha Conseil"], "ville": ["Paris"]})

    class DummyHttpClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def close(self):
            pass

        def head(self, url):
            class Resp:
                status_code = 404
                url = url

            return Resp()

    class StubProvider:
        def __init__(self, settings, http_client):
            self.http = http_client

        def search(self, query):
            return [
                site_web_search.Result(
                    url="https://alpha.fr",
                    domain="alpha.fr",
                    title="Alpha Conseil",
                    snippet="Paris",
                    rank=1,
                )
            ]

    monkeypatch.setattr(site_web_search, "HttpClient", DummyHttpClient)
    monkeypatch.setitem(site_web_search.PROVIDER_REGISTRY, "stub", StubProvider)

    enriched = site_web_search.run(df, {"providers": ["stub"]})

    assert enriched.loc[0, "site_web"] == "https://alpha.fr"
    assert enriched.loc[0, "site_web_source"].startswith("SERP")
