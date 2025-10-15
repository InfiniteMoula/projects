from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import psutil

from quality.validation import (
    FieldValidation,
    validate_email,
    validate_linkedin_url,
    validate_site_web,
    validate_telephone,
)


@dataclass(frozen=True)
class KPIField:
    """Configuration holder describing how to evaluate a given contact field."""

    metric_key: str
    column_aliases: Sequence[str]
    valid_flag_aliases: Sequence[str] = ()
    validator: Optional[Callable[[str], FieldValidation]] = None
    include_precision: bool = True
    treat_as_list: bool = False


@dataclass
class FieldMetric:
    coverage: float
    precision: Optional[float]
    present: int
    valid: int
    validatable: int


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("_", "").replace(" ", "").replace("-", "")
    return normalized.lower()


def _resolve_column(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {_normalize_key(col): col for col in columns}
    for alias in aliases:
        key = _normalize_key(alias)
        if key in lookup:
            return lookup[key]
    return None


def _coerce_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isfinite(parsed) and parsed.is_integer():
            return int(parsed)
    return None


def _extract_values(raw: object, treat_as_list: bool) -> List[str]:
    def _split_candidate(text: str) -> List[str]:
        if not text:
            return []
        if treat_as_list:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [
                    item.strip()
                    for item in (
                        str(child) for child in parsed if child is not None and str(child).strip()
                    )
                ]
            parts = [segment.strip() for segment in re.split(r"[;,]", text) if segment.strip()]
            if parts:
                return parts
        return [text]

    if raw is None:
        return []
    if raw is pd.NA:
        return []
    if isinstance(raw, float) and math.isnan(raw):
        return []
    if isinstance(raw, (list, tuple, set)):
        values: List[str] = []
        for item in raw:
            values.extend(_extract_values(item, treat_as_list))
        return [value for value in values if value]
    if isinstance(raw, dict):
        values = []
        for item in raw.values():
            values.extend(_extract_values(item, treat_as_list))
        return [value for value in values if value]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        lowered = text.lower()
        if lowered in {"nan", "none", "null"}:
            return []
        return [value for value in _split_candidate(text) if value]
    text = str(raw).strip()
    if not text:
        return []
    return [text]


def _normalize_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return None
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _validate_email(value: str) -> FieldValidation:
    return validate_email(value, check_mx=False)


FIELD_SPECS: Tuple[KPIField, ...] = (
    KPIField(
        metric_key="site_web",
        column_aliases=("site_web", "siteweb", "site", "website", "url", "Site web"),
        valid_flag_aliases=("site_web_valid", "siteweb_valid", "site_valid", "website_valid"),
        validator=validate_site_web,
    ),
    KPIField(
        metric_key="emails",
        column_aliases=(
            "email",
            "emails",
            "email_contact",
            "email_pro",
            "Email g\u00e9n\u00e9rique",
            "Email generique",
            "Email pro v\u00e9rifi\u00e9 du dirigeant",
            "email_pro_verifie_du_dirigeant",
        ),
        valid_flag_aliases=("email_valid", "emails_valid"),
        validator=_validate_email,
        treat_as_list=True,
    ),
    KPIField(
        metric_key="telephones",
        column_aliases=(
            "telephone",
            "telephones",
            "phone",
            "phones",
            "telephone_standard",
            "T\u00e9l\u00e9phone standard",
            "telephone_standard",
        ),
        valid_flag_aliases=("telephone_valid", "phone_valid", "telephone_is_valid"),
        validator=validate_telephone,
        treat_as_list=True,
    ),
    KPIField(
        metric_key="linkedin",
        column_aliases=("linkedin_url", "linkedin", "linkedin_company_url"),
        valid_flag_aliases=("linkedin_url_valid", "linkedin_valid"),
        validator=validate_linkedin_url,
        include_precision=False,
    ),
)


def compute_field_metric(df: pd.DataFrame, spec: KPIField) -> FieldMetric:
    column = _resolve_column(df.columns, spec.column_aliases)
    if column is None:
        return FieldMetric(coverage=0.0, precision=None, present=0, valid=0, validatable=0)

    series = df[column]
    total_rows = len(series)
    if total_rows == 0:
        return FieldMetric(coverage=0.0, precision=None, present=0, valid=0, validatable=0)

    valid_series = None
    if spec.valid_flag_aliases:
        valid_column = _resolve_column(df.columns, spec.valid_flag_aliases)
        if valid_column is not None:
            valid_series = df[valid_column]

    present = 0
    valid = 0
    validatable = 0

    for idx, value in series.items():
        values = _extract_values(value, spec.treat_as_list)
        if not values:
            continue

        present += 1
        is_valid: Optional[bool] = None

        if valid_series is not None:
            flag = valid_series.iloc[idx]
            is_valid = _normalize_bool(flag)

        if is_valid is None and spec.validator is not None:
            for candidate in values:
                try:
                    outcome = spec.validator(candidate)
                except Exception:
                    outcome = None
                if outcome and outcome.is_valid:
                    is_valid = True
                    break
            if is_valid is None:
                is_valid = False

        if is_valid is not None:
            validatable += 1
            if is_valid:
                valid += 1

    coverage = present / total_rows if total_rows else 0.0
    precision = (valid / validatable) if (validatable and spec.include_precision) else None
    return FieldMetric(coverage=coverage, precision=precision, present=present, valid=valid, validatable=validatable)


def iter_log_records(path: Path) -> Iterator[Dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    stripped = text.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
        return

    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def collect_numeric_metrics(obj: object, predicate: Callable[[str], bool], sink: List[int]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()
            if predicate(key_lower):
                coerced = _coerce_int(value)
                if coerced is not None:
                    sink.append(coerced)
            collect_numeric_metrics(value, predicate, sink)
    elif isinstance(obj, list):
        for item in obj:
            collect_numeric_metrics(item, predicate, sink)


def parse_logs(log_dir: Path) -> Dict[str, float]:
    http_requests = 0.0
    http_bytes = 0.0
    elapsed_seconds = 0.0
    step_duration_sum = 0.0
    status_error_events = 0
    retry_candidates: List[int] = []
    network_error_candidates: List[int] = []

    if not log_dir.exists():
        return {
            "http_requests": 0.0,
            "http_bytes": 0.0,
            "duration_s": 0.0,
            "network_errors": 0,
            "retry_count": 0,
        }

    def retry_predicate(key: str) -> bool:
        if "retry" not in key:
            return False
        if any(exclusion in key for exclusion in ("delay", "window", "strategy", "cost", "budget", "policy", "interval")):
            return False
        return True

    def network_predicate(key: str) -> bool:
        if "network" in key and "error" in key:
            return True
        if key.endswith("network_errors") or key.endswith("network_error_count"):
            return True
        if key.endswith("http_errors") or key.endswith("http_error_count"):
            return True
        return False

    for path in sorted(log_dir.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".log"}:
            continue
        for record in iter_log_records(path):
            status = str(record.get("status", "")).upper()
            if status in {"ERROR", "FAILED", "FAIL"}:
                status_error_events += 1

            duration = record.get("duration_s")
            if isinstance(duration, (int, float)) and math.isfinite(duration):
                step_duration_sum += float(duration)

            budget = record.get("budget_stats") or record.get("out", {}).get("budget_stats")
            if isinstance(budget, dict):
                http_requests = max(http_requests, float(budget.get("http_requests", 0.0) or 0.0))
                http_bytes = max(http_bytes, float(budget.get("http_bytes", 0.0) or 0.0))
                elapsed_min = budget.get("elapsed_min")
                if isinstance(elapsed_min, (int, float)) and math.isfinite(elapsed_min):
                    elapsed_seconds = max(elapsed_seconds, float(elapsed_min) * 60.0)

            collect_numeric_metrics(record, retry_predicate, retry_candidates)
            collect_numeric_metrics(record, network_predicate, network_error_candidates)

    duration_s = elapsed_seconds if elapsed_seconds else step_duration_sum
    retry_count = max(retry_candidates) if retry_candidates else 0
    network_errors = max(network_error_candidates) if network_error_candidates else status_error_events

    return {
        "http_requests": http_requests,
        "http_bytes": http_bytes,
        "duration_s": duration_s,
        "network_errors": network_errors,
        "retry_count": retry_count,
    }


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file extension for dataset: {suffix}")


def capture_resource_usage() -> Tuple[float, float]:
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    process.cpu_percent(interval=None)
    cpu_percent = process.cpu_percent(interval=0.1)
    return rss_mb, cpu_percent


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def generate_metrics(dataset_path: Path, output_path: Path) -> Dict[str, object]:
    df = load_dataframe(dataset_path)
    total_rows = len(df)

    metrics: Dict[str, object] = {
        "records_total": total_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
    }

    field_results: Dict[str, FieldMetric] = {}
    for spec in FIELD_SPECS:
        result = compute_field_metric(df, spec)
        field_results[spec.metric_key] = result
        metrics[f"coverage_{spec.metric_key}"] = round(result.coverage, 6)
        if spec.include_precision:
            metrics[f"precision_{spec.metric_key}"] = round(result.precision, 6) if result.precision is not None else None
        metrics[f"{spec.metric_key}_present"] = result.present
        metrics[f"{spec.metric_key}_valid"] = result.valid

    run_dir = dataset_path.parent
    log_metrics = parse_logs(run_dir / "logs")
    metrics.update(
        {
            "http_requests": log_metrics["http_requests"],
            "http_bytes": log_metrics["http_bytes"],
            "duration_s": log_metrics["duration_s"],
            "network_errors": log_metrics["network_errors"],
            "retry_count": log_metrics["retry_count"],
        }
    )

    duration = metrics["duration_s"]
    http_requests = metrics["http_requests"]
    if isinstance(duration, (int, float)) and duration > 0 and isinstance(http_requests, (int, float)) and http_requests > 0:
        request_rate = (float(http_requests) / float(duration)) * 60.0
    else:
        request_rate = 0.0
    metrics["requests_per_minute"] = request_rate

    rss_mb, cpu_percent = capture_resource_usage()
    metrics["memory_usage_mb"] = rss_mb
    metrics["cpu_usage_percent"] = cpu_percent

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Metrics Summary ===")
    print(f"Dataset          : {dataset_path}")
    print(f"Rows             : {total_rows:,}")
    print(f"Site web         : coverage {format_percent(field_results['site_web'].coverage)} | precision {format_percent(field_results['site_web'].precision)}")
    print(f"Emails           : coverage {format_percent(field_results['emails'].coverage)} | precision {format_percent(field_results['emails'].precision)}")
    print(f"Telephones       : coverage {format_percent(field_results['telephones'].coverage)} | precision {format_percent(field_results['telephones'].precision)}")
    print(f"LinkedIn         : coverage {format_percent(field_results['linkedin'].coverage)}")
    print(f"HTTP requests    : {int(metrics['http_requests'])}")
    print(f"Requests/minute  : {metrics['requests_per_minute']:.2f}")
    print(f"Network errors   : {metrics['network_errors']}")
    print(f"Retries          : {metrics['retry_count']}")
    print(f"Memory usage     : {metrics['memory_usage_mb']:.1f} MB")
    print(f"CPU usage        : {metrics['cpu_usage_percent']:.1f}%")
    print(f"Report written to: {output_path}")

    return metrics


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KPI metrics over an enriched dataset.")
    parser.add_argument("dataset", type=Path, help="Path to the enriched dataset (CSV or Parquet).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file (default: <run_dir>/report_metrics.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dataset_path: Path = args.dataset.resolve()
    output_path: Path
    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = dataset_path.parent / "report_metrics.json"

    try:
        generate_metrics(dataset_path, output_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
