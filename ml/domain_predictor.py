from __future__ import annotations

"""
Domain prediction model leveraging textual and structural features.

The module provides training and inference helpers to estimate the most
probable website domain for a company given a set of candidates.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .domain_features import (
    DomainFeatureConfig,
    DomainFeatureExtractor,
    coerce_candidates,
    normalize_domain,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models") / "domain_predictor.joblib"


def _first_non_empty(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif isinstance(value, (int, float)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _extract_emails(row: Mapping[str, Any]) -> List[str]:
    email_columns = ("emails", "email", "email_candidates", "emails_candidates")
    emails: List[str] = []
    for column in email_columns:
        value = row.get(column)
        if not value:
            continue
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
            emails.extend(parts)
        elif isinstance(value, Iterable):
            for item in value:
                if not item:
                    continue
                emails.append(str(item).strip())
    return [email for email in emails if "@" in email]


def _extract_postal_code(row: Mapping[str, Any]) -> str:
    return _first_non_empty(row, ("code_postal", "cp", "postal_code", "zip_code"))


def _collect_candidates(row: Mapping[str, Any]) -> List[Any]:
    candidate_columns = (
        "candidates",
        "domain_candidates",
        "site_web_candidates",
        "website_candidates",
        "serp_candidates",
    )
    candidates: List[Any] = []
    for column in candidate_columns:
        value = row.get(column)
        if value is None:
            continue
        extracted = coerce_candidates(value)
        if extracted:
            candidates.extend(extracted)
    return candidates


def _ensure_positive_candidate(candidates: List[Any], domain_true: str) -> List[Any]:
    if not domain_true:
        return candidates
    normalized = normalize_domain(domain_true)
    if not normalized:
        return candidates
    for candidate in candidates:
        if normalize_domain(candidate) == normalized:
            return candidates
    return candidates + [normalized]


def _build_feature_rows(
    df: pd.DataFrame,
    extractor: DomainFeatureExtractor,
) -> Tuple[pd.DataFrame, np.ndarray]:
    samples: List[pd.DataFrame] = []
    labels: List[np.ndarray] = []

    for _, row in df.iterrows():
        mapping = row.to_dict()
        name = _first_non_empty(
            mapping,
            ("denomination", "denomination_usuelle", "raison_sociale", "enseigne", "company_name"),
        )
        naf = _first_non_empty(mapping, ("naf", "naf_code", "ape"))
        city = _first_non_empty(mapping, ("ville", "commune", "city"))
        postal_code = _extract_postal_code(mapping)
        emails = _extract_emails(mapping)
        candidates = _collect_candidates(mapping)
        if not candidates:
            candidates = coerce_candidates(mapping.get("candidates"))
        domain_true = normalize_domain(mapping.get("domain_true") or mapping.get("domain") or mapping.get("siteweb"))
        if not candidates:
            continue
        candidates = _ensure_positive_candidate(list(candidates), domain_true)
        features = extractor.transform(
            name=name,
            naf=naf,
            city=city,
            candidates=candidates,
            emails=emails,
            postal_code=postal_code,
        )
        if features.empty:
            continue
        label = (features["normalized_domain"] == domain_true).astype(int).to_numpy(dtype=np.int8)
        samples.append(features)
        labels.append(label)

    if not samples:
        raise ValueError("No training samples could be generated for domain predictor")

    feature_frame = pd.concat(samples, ignore_index=True)
    label_array = np.concatenate(labels, axis=0)
    return feature_frame, label_array


@dataclass
class DomainPrediction:
    domain: str
    probability: float
    features: Dict[str, float]


class DomainPredictor:
    """Wrapper around the trained model artifact."""

    def __init__(
        self,
        *,
        model: LogisticRegression,
        scaler: StandardScaler,
        feature_columns: Sequence[str],
        extractor_artifact: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self.model = model
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.extractor = DomainFeatureExtractor.from_artifact(extractor_artifact)
        self.metadata = metadata

    def predict(
        self,
        *,
        name: str,
        naf: str,
        city: str,
        candidates: Sequence[Any],
        emails: Optional[Sequence[str]] = None,
        postal_code: Optional[str] = None,
    ) -> List[DomainPrediction]:
        if not candidates:
            return []
        feature_frame = self.extractor.transform(
            name=name,
            naf=naf,
            city=city,
            candidates=candidates,
            emails=emails,
            postal_code=postal_code,
        )
        if feature_frame.empty:
            return []
        domains = feature_frame["normalized_domain"].tolist()
        features = feature_frame.copy()
        missing = [col for col in self.feature_columns if col not in features.columns]
        for column in missing:
            features[column] = 0.0
        features = features[self.feature_columns]
        matrix = features.to_numpy(dtype=np.float32)
        matrix_scaled = self.scaler.transform(matrix)
        probabilities = self.model.predict_proba(matrix_scaled)[:, 1]
        predictions: List[DomainPrediction] = []
        for idx in range(len(features)):
            predictions.append(
                DomainPrediction(
                    domain=domains[idx],
                    probability=float(probabilities[idx]),
                    features={col: float(features.iloc[idx][col]) for col in self.feature_columns},
                )
            )
        predictions.sort(key=lambda item: item.probability, reverse=True)
        return predictions

    def predict_best(
        self,
        *,
        name: str,
        naf: str,
        city: str,
        candidates: Sequence[Any],
        emails: Optional[Sequence[str]] = None,
        postal_code: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        results = self.predict(
            name=name,
            naf=naf,
            city=city,
            candidates=candidates,
            emails=emails,
            postal_code=postal_code,
        )
        if not results:
            return None, 0.0
        return results[0].domain, results[0].probability


def _prepare_features(
    df: pd.DataFrame,
    extractor: DomainFeatureExtractor,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    feature_frame, labels = _build_feature_rows(df, extractor)
    base_columns = {"domain", "normalized_domain", "core_domain", "suffix"}
    feature_columns = [col for col in feature_frame.columns if col not in base_columns]
    if not feature_columns:
        raise ValueError("No numeric features available for training")
    data = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    return feature_frame, labels, feature_columns


def train_domain_predictor(
    df: pd.DataFrame,
    *,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    config: Optional[DomainFeatureConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> DomainPredictor:
    """
    Train the domain predictor on the provided dataset and persist the artifact.
    """
    if df.empty:
        raise ValueError("Training dataset is empty")

    logger = logger or log
    extractor = DomainFeatureExtractor(config=config)
    extractor.fit_vectorizer(df)

    feature_frame, labels, feature_columns = _prepare_features(df, extractor)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0:
        raise ValueError("Training dataset contains no positive samples")
    if negatives == 0:
        raise ValueError("Training dataset contains no negative samples")
    scaler = StandardScaler()
    X = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_scaled, labels)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except ValueError:
        roc_auc = float("nan")
    try:
        avg_precision = average_precision_score(labels, probabilities)
    except ValueError:
        avg_precision = float("nan")

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "extractor": extractor.to_artifact(),
        "metadata": {
            "training_samples": int(len(labels)),
            "positive_samples": positives,
            "negative_samples": negatives,
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
        },
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    _PREDICTOR_CACHE["model"] = None
    _PREDICTOR_CACHE["checked"] = False

    logger.info(
        "Domain predictor trained | samples=%d | positives=%d | accuracy=%.3f | roc_auc=%.3f | avg_precision=%.3f",
        artifact["metadata"]["training_samples"],
        artifact["metadata"]["positive_samples"],
        artifact["metadata"]["accuracy"],
        artifact["metadata"]["roc_auc"],
        artifact["metadata"]["average_precision"],
    )

    return DomainPredictor(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        extractor_artifact=artifact["extractor"],
        metadata=artifact["metadata"],
    )


_PREDICTOR_CACHE: Dict[str, Any] = {"model": None, "checked": False}


def load_domain_predictor(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Optional[DomainPredictor]:
    cached = _PREDICTOR_CACHE.get("model")
    if cached is not None:
        return cached
    if _PREDICTOR_CACHE.get("checked"):
        return None
    model_path = Path(model_path)
    if not model_path.exists():
        log.debug("Domain predictor artifact missing at %s", model_path)
        _PREDICTOR_CACHE["checked"] = True
        return None
    artifact = joblib.load(model_path)
    predictor = DomainPredictor(
        model=artifact["model"],
        scaler=artifact["scaler"],
        feature_columns=artifact["feature_columns"],
        extractor_artifact=artifact["extractor"],
        metadata=artifact["metadata"],
    )
    _PREDICTOR_CACHE["model"] = predictor
    _PREDICTOR_CACHE["checked"] = True
    return predictor


def predict_best_domain(
    name: str,
    naf: str,
    city: str,
    candidates: Sequence[Any],
    *,
    postal_code: Optional[str] = None,
    emails: Optional[Sequence[str]] = None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Tuple[Optional[str], float]:
    """
    Predict the most likely domain from ``candidates``.
    """
    predictor = load_domain_predictor(model_path=model_path)
    if predictor is None:
        return None, 0.0
    domain, probability = predictor.predict_best(
        name=name or "",
        naf=naf or "",
        city=city or "",
        candidates=list(candidates),
        emails=emails,
        postal_code=postal_code,
    )
    return domain, float(probability)


__all__ = [
    "DomainPrediction",
    "DomainPredictor",
    "DEFAULT_MODEL_PATH",
    "load_domain_predictor",
    "predict_best_domain",
    "train_domain_predictor",
]
