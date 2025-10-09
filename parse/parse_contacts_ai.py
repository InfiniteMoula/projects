"""Experimental OCR enhancement for contacts using pytesseract if available."""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from utils import io as io_utils

try:  # pragma: no cover
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore

try:  # pragma: no cover
    from lxml import html as lxml_html  # type: ignore
except ImportError:  # pragma: no cover
    lxml_html = None  # type: ignore


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{7,}")


def _load_no_contact(outdir: Path) -> pd.DataFrame:
    path = outdir / "contacts" / "no_contact.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_dynamic_pages(outdir: Path) -> pd.DataFrame:
    candidates = [
        outdir / "headless" / "pages_dynamic.parquet",
        outdir / "crawl" / "pages.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                continue
    return pd.DataFrame()


def _extract_images_from_html(row: pd.Series) -> List[bytes]:
    html = str(row.get("content_html") or row.get("content_html_trunc") or "")
    images: List[bytes] = []
    if not html:
        return images
    if lxml_html is None:
        return images

    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return images

    for img in tree.xpath("//img[@src]"):
        src = img.get("src") or ""
        if not src:
            continue
        if src.startswith("data:image"):
            try:
                header, data = src.split(",", 1)
                images.append(base64.b64decode(data))
            except Exception:
                continue
        else:
            candidate = str(row.get("local_path") or "")
            if candidate and Path(candidate).exists():
                try:
                    images.append(Path(candidate).read_bytes())
                except Exception:
                    continue
    return images


def _extract_text_from_image(data: bytes) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return pytesseract.image_to_string(img)  # type: ignore[attr-defined]
    except Exception:
        return ""


def _extract_contacts(text: str) -> Dict[str, Set[str]]:
    emails = {match.lower() for match in EMAIL_REGEX.findall(text)}
    phones = {re.sub(r"\s+", " ", match).strip() for match in PHONE_REGEX.findall(text)}
    return {"emails": emails, "phones": phones}


def run(cfg: dict, ctx: dict) -> dict:
    if pytesseract is None:  # pragma: no cover
        return {"status": "SKIPPED", "reason": "PYTESSERACT_MISSING"}

    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    no_contact_df = _load_no_contact(outdir)
    if no_contact_df.empty:
        return {"status": "SKIPPED", "reason": "NO_TARGETS"}

    pages_df = _load_dynamic_pages(outdir)
    if pages_df.empty:
        return {"status": "SKIPPED", "reason": "NO_PAGES"}

    pages_df["domain"] = pages_df.get("domain", "").astype(str)
    domain_map = pages_df.groupby("domain")

    results: List[Dict[str, object]] = []
    for _, company in no_contact_df.iterrows():
        domain = str(company.get("domain") or "").strip().lower()
        siren = str(company.get("siren") or "").strip()
        if not domain:
            continue
        group = domain_map.get_group(domain) if domain in domain_map.groups else pd.DataFrame()
        if group.empty:
            continue

        emails: Set[str] = set()
        phones: Set[str] = set()
        for _, page in group.iterrows():
            images = _extract_images_from_html(page)
            for blob in images:
                text = _extract_text_from_image(blob)
                if not text:
                    continue
                contacts = _extract_contacts(text)
                emails.update(contacts["emails"])
                phones.update(contacts["phones"])

        if emails or phones:
            results.append(
                {
                    "siren": siren or None,
                    "domain": domain,
                    "emails": sorted(emails),
                    "phones": sorted(phones),
                }
            )

    if not results:
        return {"status": "SKIPPED", "reason": "NO_FINDINGS"}

    contacts_dir = io_utils.ensure_dir(outdir / "contacts")
    output_path = contacts_dir / "contacts_ai.parquet"
    pd.DataFrame(results).to_parquet(output_path, index=False)

    return {
        "status": "OK",
        "records": len(results),
        "output": str(output_path),
    }


__all__ = ["run"]
