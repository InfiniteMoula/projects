"""LLM-backed contact extraction stub.

This module exposes a small interface that allows plugging a large language model
based extractor for contact information contained in HTML snippets.  The default
implementation is a no-op stub returning empty results which keeps the system
behaviour unchanged when the feature is disabled.  A custom extractor can be
registered via :func:`set_extractor` and will be invoked by
:func:`extract_contacts`.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

Extractor = Callable[[str, Mapping[str, object]], Mapping[str, Sequence[str] | str | Iterable[str]]]


_registered_extractor: Optional[Extractor] = None


def set_extractor(extractor: Optional[Extractor]) -> None:
    """Register a callable used to perform contact extraction.

    Passing ``None`` restores the default stub implementation.  The callable
    receives the HTML payload as a string along with optional hints and must
    return a mapping containing ``emails``, ``phones`` and ``linkedin`` keys.
    Each value should be an iterable of strings.
    """

    global _registered_extractor
    _registered_extractor = extractor


def _normalize_output(value: Mapping[str, object] | Sequence[str] | str | None) -> Mapping[str, Sequence[str]]:
    if isinstance(value, Mapping):
        payload: MutableMapping[str, Sequence[str]] = {}
        for key in ("emails", "phones", "linkedin"):
            payload[key] = _coerce_iterable(value.get(key))
        return payload
    if isinstance(value, (str, bytes)):
        return {"emails": [str(value)], "phones": [], "linkedin": []}
    if isinstance(value, Sequence):
        return {"emails": [str(item) for item in value], "phones": [], "linkedin": []}
    return {"emails": [], "phones": [], "linkedin": []}


def _coerce_iterable(value: object) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        items = []
        for _, candidate in value.items():
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                items.append(text)
        return items
    if isinstance(value, Iterable):
        items = []
        for candidate in value:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def extract_contacts(html: str, hints: Optional[Mapping[str, object]] = None) -> Dict[str, Sequence[str]]:
    """Extract contact information from *html*.

    Parameters
    ----------
    html:
        Raw HTML payload to inspect.
    hints:
        Optional mapping containing metadata about the page being processed.

    Returns
    -------
    dict
        A mapping with ``emails``, ``phones`` and ``linkedin`` keys containing
        sequences of strings.
    """

    extractor = _registered_extractor
    if extractor is None:
        return {"emails": [], "phones": [], "linkedin": []}

    payload = extractor(html or "", hints or {})
    if not isinstance(payload, Mapping):
        return {"emails": [], "phones": [], "linkedin": []}
    normalised = _normalize_output(payload)
    return {
        "emails": list(normalised.get("emails", [])),
        "phones": list(normalised.get("phones", [])),
        "linkedin": list(normalised.get("linkedin", [])),
    }


__all__ = ["extract_contacts", "set_extractor"]
