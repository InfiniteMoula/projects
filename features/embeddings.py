"""Text embedding utilities for semantic comparisons."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from sklearn.feature_extraction.text import HashingVectorizer

LOGGER = logging.getLogger("features.embeddings")

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_HASH_VECTORIZER = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    norm=None,
    lowercase=True,
    stop_words="english",
)

_SENTENCE_MODELS: dict[str, SentenceTransformer] = {}


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def generate_text(company_row: Mapping[str, object] | Sequence | object) -> str:
    """Generate a descriptive text for a company row."""

    def _get(row: object, keys: Iterable[str]) -> str:
        for key in keys:
            value = None
            if isinstance(row, Mapping):
                value = row.get(key)
            if value is None and hasattr(row, "get"):
                try:
                    value = row.get(key)  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive
                    value = None
            if value is None and hasattr(row, key):
                value = getattr(row, key, None)
            text = _coerce_text(value)
            if text:
                return text
        return ""

    name = _get(company_row, ("denomination", "raison_sociale", "enseigne", "nom", "name"))
    city = _get(company_row, ("ville", "commune", "city"))
    naf_code = _get(company_row, ("naf_code", "code_naf", "ape", "naf"))
    naf_label = _get(company_row, ("naf_label", "libelle_naf", "libellenomenclatureactiviteprincipaleetablissement"))

    parts = []
    if name:
        parts.append(name)
    if city:
        parts.append(city)

    naf_parts = [segment for segment in (naf_code, naf_label) if segment]
    if naf_parts:
        parts.append("NAF " + " - ".join(naf_parts))

    return " \n ".join(parts).strip()


def _load_sentence_model(model_name: str) -> SentenceTransformer | None:
    if SentenceTransformer is None:
        return None
    cached = _SENTENCE_MODELS.get(model_name)
    if cached is not None:
        return cached
    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Failed to load SentenceTransformer %s: %s", model_name, exc)
        return None
    _SENTENCE_MODELS[model_name] = model
    return model


def _embed_with_model(text: str, model_name: str) -> np.ndarray | None:
    model = _load_sentence_model(model_name)
    if model is None:
        return None
    embedding = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    if embedding is None or len(embedding) == 0:
        return None
    vector = np.asarray(embedding[0], dtype=np.float32)
    return vector


def _embed_with_hashing(text: str) -> np.ndarray:
    vec = _HASH_VECTORIZER.transform([text])
    dense = vec.toarray()[0].astype(np.float32, copy=False)
    norm = float(np.linalg.norm(dense))
    if norm > 0:
        dense /= norm
    return dense


@lru_cache(maxsize=2048)
def _cached_embedding(text: str, model_name: str) -> tuple[float, ...]:
    if not text:
        return tuple()
    vector = _embed_with_model(text, model_name)
    if vector is None:
        vector = _embed_with_hashing(text)
    return tuple(float(x) for x in vector.tolist())


def embed(text: str, *, model_name: str | None = None) -> np.ndarray:
    """Compute an embedding for *text*."""

    clean = _coerce_text(text)
    if not clean:
        return np.zeros(_HASH_VECTORIZER.n_features, dtype=np.float32)
    name = model_name or DEFAULT_MODEL
    data = _cached_embedding(clean, name)
    if not data:
        return np.zeros(_HASH_VECTORIZER.n_features, dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors *a* and *b*."""

    if a is None or b is None:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


__all__ = ["DEFAULT_MODEL", "cosine", "embed", "generate_text"]
