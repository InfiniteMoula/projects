"""Budget tracking and enforcement middleware for pipeline steps."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from config.budget_config import get_budget_thresholds

LOGGER = logging.getLogger("utils.budget_middleware")
BUDGET_DEFAULTS = get_budget_thresholds()


def _resolve_budget_limit(value: Any, default: Optional[int]) -> int:
    """Return an integer budget limit falling back to *default* when necessary."""

    if value is None:
        return int(default or 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default or 0)


class BudgetExceededError(RuntimeError):
    """Raised when a budget limit is exceeded."""


@dataclass
class BudgetTracker:
    """Tracks resource usage against configured budgets."""
    
    # Budget limits from job config
    max_http_requests: int = 0
    max_http_bytes: int = 0
    time_budget_min: int = 0
    ram_mb: int = 0
    
    # Current usage counters
    http_requests: int = field(default=0, init=False)
    http_bytes: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    
    def __post_init__(self):
        """Initialize tracker."""
        self.start_time = time.time()
        LOGGER.info(
            "Budget tracker initialized: max_requests=%d, max_bytes=%d, time_budget=%d min, ram=%d MB",
            self.max_http_requests, self.max_http_bytes, self.time_budget_min, self.ram_mb
        )
    
    def track_http_request(self, response_bytes: int = 0) -> None:
        """Track an HTTP request and its response size."""
        self.http_requests += 1
        self.http_bytes += response_bytes
        
        LOGGER.debug(
            "HTTP request tracked: count=%d/%d, bytes=%d/%d", 
            self.http_requests, self.max_http_requests,
            self.http_bytes, self.max_http_bytes
        )
        
        # Check budget limits
        self._check_http_limits()
    
    def _check_http_limits(self) -> None:
        """Check if HTTP request/bytes budgets are exceeded."""
        if self.max_http_requests > 0 and self.http_requests > self.max_http_requests:
            raise BudgetExceededError(
                f"HTTP request budget exceeded: {self.http_requests} > {self.max_http_requests}"
            )
        
        if self.max_http_bytes > 0 and self.http_bytes > self.max_http_bytes:
            raise BudgetExceededError(
                f"HTTP bytes budget exceeded: {self.http_bytes} > {self.max_http_bytes}"
            )
    
    def check_time_budget(self) -> None:
        """Check if time budget is exceeded."""
        if self.time_budget_min <= 0:
            return
            
        elapsed_min = (time.time() - self.start_time) / 60.0
        if elapsed_min > self.time_budget_min:
            raise BudgetExceededError(
                f"Time budget exceeded: {elapsed_min:.1f} min > {self.time_budget_min} min"
            )
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        elapsed_min = (time.time() - self.start_time) / 60.0
        return {
            "http_requests": self.http_requests,
            "max_http_requests": self.max_http_requests,
            "http_bytes": self.http_bytes,
            "max_http_bytes": self.max_http_bytes,
            "elapsed_min": round(elapsed_min, 2),
            "time_budget_min": self.time_budget_min,
            "http_requests_pct": (self.http_requests / max(1, self.max_http_requests)) * 100 if self.max_http_requests > 0 else 0,
            "http_bytes_pct": (self.http_bytes / max(1, self.max_http_bytes)) * 100 if self.max_http_bytes > 0 else 0,
            "time_budget_pct": (elapsed_min / max(1, self.time_budget_min)) * 100 if self.time_budget_min > 0 else 0,
        }


@dataclass 
class KPICalculator:
    """Calculates and compares final KPIs against targets."""
    
    # KPI targets from job config
    min_quality_score: float = 0.0
    max_dup_pct: float = 100.0
    min_url_valid_pct: float = 0.0
    min_domain_resolved_pct: float = 0.0
    min_email_plausible_pct: float = 0.0
    min_lines_per_s: float = 0.0
    
    def calculate_final_kpis(self, context: Dict[str, Any], results: list) -> Dict[str, Any]:
        """Calculate final KPI values from pipeline results."""
        # Extract metrics from step results
        total_duration = sum(r.get("duration_s", 0) for r in results)
        total_lines = context.get("total_lines_processed")
        if not isinstance(total_lines, (int, float)):
            total_lines = 0

        if not total_lines:
            fallback_total = 0
            for result in results:
                out = result.get("out")
                if isinstance(out, dict):
                    rows_written = out.get("rows_written")
                    if isinstance(rows_written, (int, float)):
                        fallback_total += int(rows_written)
            if fallback_total:
                total_lines = fallback_total
            else:
                rows_from_context = context.get("rows_written")
                if isinstance(rows_from_context, (int, float)):
                    total_lines = int(rows_from_context)

        # Calculate actual KPI values
        actual_kpis = {
            "quality_score": self._extract_quality_score(results),
            "dup_pct": self._extract_duplicate_percentage(results),
            "url_valid_pct": self._extract_url_valid_percentage(results),
            "domain_resolved_pct": self._extract_domain_resolved_percentage(results),
            "email_plausible_pct": self._extract_email_plausible_percentage(results),
            "lines_per_s": total_lines / max(1, total_duration),
        }
        
        # Compare against targets
        kpi_comparison = {
            "quality_score_met": actual_kpis["quality_score"] >= self.min_quality_score,
            "dup_pct_met": actual_kpis["dup_pct"] <= self.max_dup_pct,
            "url_valid_pct_met": actual_kpis["url_valid_pct"] >= self.min_url_valid_pct,
            "domain_resolved_pct_met": actual_kpis["domain_resolved_pct"] >= self.min_domain_resolved_pct,
            "email_plausible_pct_met": actual_kpis["email_plausible_pct"] >= self.min_email_plausible_pct,
            "lines_per_s_met": actual_kpis["lines_per_s"] >= self.min_lines_per_s,
        }
        
        # Overall KPI success
        all_kpis_met = all(kpi_comparison.values())
        
        return {
            "actual_kpis": actual_kpis,
            "target_kpis": {
                "min_quality_score": self.min_quality_score,
                "max_dup_pct": self.max_dup_pct,
                "min_url_valid_pct": self.min_url_valid_pct,
                "min_domain_resolved_pct": self.min_domain_resolved_pct,
                "min_email_plausible_pct": self.min_email_plausible_pct,
                "min_lines_per_s": self.min_lines_per_s,
            },
            "kpi_comparison": kpi_comparison,
            "all_kpis_met": all_kpis_met,
        }
    
    def _extract_quality_score(self, results: list) -> float:
        """Extract quality score from quality.score step results."""
        for result in results:
            if result.get("step") == "quality.score":
                return result.get("out", {}).get("avg_quality_score", 0.0)
        return 0.0
    
    def _extract_duplicate_percentage(self, results: list) -> float:
        """Extract duplicate percentage from quality.dedupe step results."""
        for result in results:
            if result.get("step") == "quality.dedupe":
                out = result.get("out", {})
                # Try new format first, fallback to old format for compatibility
                if "duplicate_rate_pct" in out:
                    return out.get("duplicate_rate_pct", 0.0)
                else:
                    # Legacy format calculation
                    total = out.get("total_records", out.get("before", 1))
                    duplicates = out.get("duplicates_removed", 0)
                    return (duplicates / max(1, total)) * 100
        return 0.0
    
    def _extract_url_valid_percentage(self, results: list) -> float:
        """Extract URL validation percentage from HTTP collection steps."""
        for result in results:
            if result.get("step") in ["http.static", "http.sitemap"]:
                out = result.get("out", {})
                total_urls = out.get("total_urls", 1)
                valid_urls = out.get("valid_urls", 0)
                return (valid_urls / max(1, total_urls)) * 100
        return 0.0
    
    def _extract_domain_resolved_percentage(self, results: list) -> float:
        """Extract domain resolution percentage from DNS checks."""
        for result in results:
            if result.get("step") == "enrich.dns":
                out = result.get("out", {})
                total_domains = out.get("total_domains", 1)
                resolved_domains = out.get("resolved_domains", 0)
                return (resolved_domains / max(1, total_domains)) * 100
        return 0.0
    
    def _extract_email_plausible_percentage(self, results: list) -> float:
        """Extract email plausibility percentage from email heuristics."""
        for result in results:
            if result.get("step") == "enrich.email":
                out = result.get("out", {})
                total_emails = out.get("total_emails", 1)
                plausible_emails = out.get("plausible_emails", 0)
                return (plausible_emails / max(1, total_emails)) * 100
        return 0.0


def create_budget_tracker(job_config: Dict[str, Any]) -> Optional[BudgetTracker]:
    """Create a budget tracker from job configuration."""
    budgets_cfg = job_config.get("budgets") or {}
    if not budgets_cfg:
        LOGGER.info("No budgets configured, skipping budget tracking")
        return None
    if not isinstance(budgets_cfg, dict):
        LOGGER.warning("Invalid budgets configuration type: %s", type(budgets_cfg))
        budgets_cfg = {}

    max_http_requests = _resolve_budget_limit(
        budgets_cfg.get("max_http_requests"), BUDGET_DEFAULTS.max_http_requests
    )
    max_http_bytes = _resolve_budget_limit(
        budgets_cfg.get("max_http_bytes"), BUDGET_DEFAULTS.max_http_bytes
    )
    time_budget_min = _resolve_budget_limit(
        budgets_cfg.get("time_budget_min"), BUDGET_DEFAULTS.time_budget_min
    )
    ram_mb = _resolve_budget_limit(budgets_cfg.get("ram_mb"), BUDGET_DEFAULTS.ram_mb)

    if not any([max_http_requests, max_http_bytes, time_budget_min, ram_mb]):
        LOGGER.info("No budgets configured, skipping budget tracking")
        return None

    return BudgetTracker(
        max_http_requests=max_http_requests,
        max_http_bytes=max_http_bytes,
        time_budget_min=time_budget_min,
        ram_mb=ram_mb,
    )


def create_kpi_calculator(job_config: Dict[str, Any]) -> Optional[KPICalculator]:
    """Create a KPI calculator from job configuration."""
    kpi_targets = job_config.get("kpi_targets", {})
    
    if not kpi_targets:
        LOGGER.info("No KPI targets configured, skipping KPI calculation")
        return None
    
    return KPICalculator(
        min_quality_score=kpi_targets.get("min_quality_score", 0.0),
        max_dup_pct=kpi_targets.get("max_dup_pct", 100.0),
        min_url_valid_pct=kpi_targets.get("min_url_valid_pct", 0.0),
        min_domain_resolved_pct=kpi_targets.get("min_domain_resolved_pct", 0.0),
        min_email_plausible_pct=kpi_targets.get("min_email_plausible_pct", 0.0),
        min_lines_per_s=kpi_targets.get("min_lines_per_s", 0.0),
    )


@contextmanager
def http_request_tracking(budget_tracker: Optional[BudgetTracker]):
    """Context manager to track HTTP requests with budget enforcement."""
    if budget_tracker is None:
        yield lambda size=0: None  # No-op if no tracker
        return
    
    def track_request(response_size: int = 0):
        budget_tracker.track_http_request(response_size)
    
    yield track_request