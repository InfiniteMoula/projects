import httpx

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
