from __future__ import annotations

import httpx
import pytest

from nethttp.internal_crawler import CrawlResult, PageResult, crawl_internal_pages


class MockSite:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url.copy_with(fragment=None))
        body = self.pages.get(url)
        if body is None and url.endswith("example.com"):
            body = self.pages.get(f"{url}/")
        if body is None:
            return httpx.Response(404, text="not found", headers={"content-type": "text/html"})
        return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


@pytest.fixture()
def mock_client() -> httpx.Client:
    site = MockSite(
        {
            "https://example.com/": """
                <html>
                    <body>
                        <p>Bienvenue</p>
                        <a href='/team'>Team</a>
                        <a href='/contact'>Contact</a>
                        <a href='https://external.test/about'>External</a>
                    </body>
                </html>
            """,
            "https://example.com/contact": """
                <html>
                    <body>
                        <p>Contactez-nous: contact@example.com</p>
                        <a href='/mentions-legales'>Mentions</a>
                    </body>
                </html>
            """,
            "https://example.com/team": """
                <html>
                    <body>
                        <p>Equipe: equipe@example.com</p>
                    </body>
                </html>
            """,
            "https://example.com/mentions-legales": """
                <html>
                    <body>
                        <p>Mentions légales</p>
                    </body>
                </html>
            """,
            "https://external.test/about": "<html><body>Should not fetch</body></html>",
        }
    )
    transport = httpx.MockTransport(site)
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        yield client


def test_crawl_prioritises_keyword_pages(mock_client: httpx.Client) -> None:
    result = crawl_internal_pages("https://example.com", 4, client=mock_client)
    assert isinstance(result, CrawlResult)
    assert len(result.pages) == 4

    urls = [page.url for page in result.pages]
    assert urls[0] in {"https://example.com", "https://example.com/"}
    # Contact page should appear before the team page thanks to keyword priority.
    assert urls[1].endswith("/contact")
    assert urls[2].endswith("/mentions-legales")
    assert urls[3].endswith("/team")

    aggregated_emails = set(result.emails)
    assert aggregated_emails == {"contact@example.com", "equipe@example.com"}

    contact_page = next(page for page in result.pages if page.url.endswith("/contact"))
    assert isinstance(contact_page, PageResult)
    assert "contact@example.com" in contact_page.emails


def test_crawl_stops_at_max_pages(mock_client: httpx.Client) -> None:
    result = crawl_internal_pages("https://example.com", 2, client=mock_client)
    assert len(result.pages) == 2
    assert all(page.url.startswith("https://example.com") for page in result.pages)


def test_invalid_url_raises() -> None:
    with pytest.raises(ValueError):
        crawl_internal_pages("/relative", 1)


def test_zero_max_pages_returns_empty() -> None:
    result = crawl_internal_pages("https://example.com", 0)
    assert result.pages == tuple()
    assert result.emails == tuple()
