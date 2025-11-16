"""Download images containing 'email' indications from a webpage and run OCR."""
from __future__ import annotations

import argparse
import re
from io import BytesIO
import logging
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from PIL import Image
import pytesseract

from proxy_manager import ProxyManager


EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
logger = logging.getLogger(__name__)
PROXY_MANAGER = ProxyManager()


class DownloadError(RuntimeError):
    """Raised when downloading the HTML page or an image fails."""


def fetch_html(url: str) -> str:
    try:
        kwargs = {"timeout": 15}
        proxies = PROXY_MANAGER.as_requests()
        if proxies:
            kwargs["proxies"] = proxies
        response = requests.get(url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        if PROXY_MANAGER.enabled:
            logger.warning("Proxy HTTP GET failed for %s: %s", url, exc)
        raise DownloadError(f"Unable to fetch HTML from {url}: {exc}") from exc
    return response.text


def parse_image_elements(html: str) -> Iterable[Tag]:
    soup = BeautifulSoup(html, "html.parser")
    return soup.find_all("img")


def is_email_related(src: str, alt: Optional[str], title: Optional[str]) -> bool:
    values: List[str] = []
    if src:
        filename = Path(urlparse(src).path).name
        values.append(filename)
    if alt:
        values.append(alt)
    if title:
        values.append(title)

    return any("email" in value.lower() for value in values if value)


def download_image(img_url: str) -> bytes:
    try:
        kwargs = {"timeout": 15}
        proxies = PROXY_MANAGER.as_requests()
        if proxies:
            kwargs["proxies"] = proxies
        response = requests.get(img_url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        if PROXY_MANAGER.enabled:
            logger.warning("Proxy image download failed for %s: %s", img_url, exc)
        raise DownloadError(f"Failed to download image {img_url}: {exc}") from exc
    return response.content


def guess_filename(img_url: str, download_dir: Path, counter: int) -> Path:
    parsed = urlparse(img_url)
    filename = Path(parsed.path).name
    if not filename:
        filename = f"image_{counter}.png"

    destination = download_dir / filename
    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix or ".png"
        destination = download_dir / f"{stem}_{counter}{suffix}"

    return destination


def extract_email_from_image(image_bytes: bytes) -> Optional[str]:
    with Image.open(BytesIO(image_bytes)) as image:
        text = pytesseract.image_to_string(image)
    match = EMAIL_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def process_page(url: str, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    html = fetch_html(url)
    images = parse_image_elements(html)

    base_url = url
    matches = []
    for counter, img in enumerate(images, start=1):
        src = img.get("src")
        alt = img.get("alt")
        title = img.get("title")
        if not src:
            continue
        if not is_email_related(src, alt, title):
            continue

        full_url = urljoin(base_url, src)
        try:
            image_bytes = download_image(full_url)
        except DownloadError as exc:
            print(exc)
            continue

        destination = guess_filename(full_url, download_dir, counter)
        destination.write_bytes(image_bytes)
        print(f"Downloaded {full_url} -> {destination}")

        email = extract_email_from_image(image_bytes)
        if email:
            matches.append((destination, email))
            print(f"Potential email found in {destination}: {email}")
        else:
            print(f"No email detected in {destination}")

    if not matches:
        print("No emails detected in email-related images.")
    else:
        print("Summary of detected emails:")
        for path, email in matches:
            print(f"  {path}: {email}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL of the webpage to scan")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_email_images"),
        help="Directory where images will be saved",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        process_page(args.url, args.output_dir)
    except DownloadError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
