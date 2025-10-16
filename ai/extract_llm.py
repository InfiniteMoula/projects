"""Interfaces for contact extraction performed by local language models.

This module currently exposes a stubbed implementation that returns empty
results. The callable signature is stable so that future, fully-fledged LLM
integrations can plug into the enrichment pipeline without touching callers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping

LOGGER = logging.getLogger("ai.extract_llm")

ContactExtraction = Dict[str, List[str]]


def extract_contacts(html: str, hints: Mapping[str, object] | None = None) -> ContactExtraction:
    """Extract contact information from *html* using a language model.

    Parameters
    ----------
    html:
        Raw HTML content to analyse.
    hints:
        Optional metadata that can guide the extraction process (such as URL or
        company domain). The stub ignores these hints but validates the
        structure to ensure forwards compatibility.

    Returns
    -------
    dict
        A dictionary containing three keys: ``emails``, ``phones`` and
        ``linkedin``. Each maps to a list of strings. The default stub returns
        empty lists so that callers can rely on consistent typing while waiting
        for the real LLM-backed implementation.
    """

    if not isinstance(html, str):
        LOGGER.debug("extract_contacts received non-string html: %r", type(html))
        html = ""

    if hints is None:
        hints = {}
    elif not isinstance(hints, Mapping):  # pragma: no cover - defensive
        LOGGER.debug("extract_contacts received invalid hints type: %r", type(hints))
        hints = {}
    else:
        # Trigger materialisation to surface unexpected lazy containers early.
        hints = {str(key): value for key, value in hints.items()}

    return {
        "emails": [],
        "phones": [],
        "linkedin": [],
    }


__all__ = ["extract_contacts", "ContactExtraction"]
