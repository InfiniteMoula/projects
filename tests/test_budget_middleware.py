"""Tests for budget middleware integration."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from utils.budget_middleware import (
    BudgetTracker, 
    KPICalculator, 
    BudgetExceededError,
    create_budget_tracker,
    create_kpi_calculator
)


class TestBudgetTracker:
    """Test budget tracking functionality."""

    def test_create_budget_tracker_with_config(self):
        """Test creating budget tracker from job config."""
        job_config = {
            "budgets": {
                "max_http_requests": 100,
                "max_http_bytes": 1000000,
                "time_budget_min": 30,
                "ram_mb": 2048
            }
        }
        
        tracker = create_budget_tracker(job_config)
        assert tracker is not None
        assert tracker.max_http_requests == 100
        assert tracker.max_http_bytes == 1000000
        assert tracker.time_budget_min == 30
        assert tracker.ram_mb == 2048

    def test_create_budget_tracker_no_config(self):
        """Test creating budget tracker with no budget config."""
        job_config = {}
        
        tracker = create_budget_tracker(job_config)
        assert tracker is None

    def test_http_request_tracking(self):
        """Test HTTP request tracking and enforcement."""
        tracker = BudgetTracker(
            max_http_requests=3,
            max_http_bytes=1000
        )
        
        # Track requests within limits
        tracker.track_http_request(200)
        tracker.track_http_request(300)
        tracker.track_http_request(100)
        
        stats = tracker.get_current_stats()
        assert stats["http_requests"] == 3
        assert stats["http_bytes"] == 600
        assert stats["http_requests_pct"] == 100.0
        assert stats["http_bytes_pct"] == 60.0
        
        # Exceeding request limit should raise error
        with pytest.raises(BudgetExceededError, match="HTTP request budget exceeded"):
            tracker.track_http_request(100)

    def test_http_bytes_limit(self):
        """Test HTTP bytes limit enforcement."""
        tracker = BudgetTracker(
            max_http_requests=10,
            max_http_bytes=500
        )
        
        # Within limits
        tracker.track_http_request(400)
        
        # Exceeding bytes limit should raise error
        with pytest.raises(BudgetExceededError, match="HTTP bytes budget exceeded"):
            tracker.track_http_request(200)

    def test_time_budget_check(self):
        """Test time budget checking."""
        tracker = BudgetTracker(time_budget_min=0.001)  # Very short budget
        
        import time
        time.sleep(0.1)  # Wait a bit to exceed budget
        
        with pytest.raises(BudgetExceededError, match="Time budget exceeded"):
            tracker.check_time_budget()


class TestKPICalculator:
    """Test KPI calculation functionality."""

    def test_create_kpi_calculator_with_config(self):
        """Test creating KPI calculator from job config."""
        job_config = {
            "kpi_targets": {
                "min_quality_score": 80,
                "max_dup_pct": 2.0,
                "min_url_valid_pct": 85,
                "min_domain_resolved_pct": 75,
                "min_email_plausible_pct": 60,
                "min_lines_per_s": 50
            }
        }
        
        calculator = create_kpi_calculator(job_config)
        assert calculator is not None
        assert calculator.min_quality_score == 80
        assert calculator.max_dup_pct == 2.0
        assert calculator.min_url_valid_pct == 85

    def test_create_kpi_calculator_no_config(self):
        """Test creating KPI calculator with no KPI config."""
        job_config = {}
        
        calculator = create_kpi_calculator(job_config)
        assert calculator is None

    def test_kpi_calculation(self):
        """Test KPI calculation from pipeline results."""
        calculator = KPICalculator(
            min_quality_score=80,
            max_dup_pct=2.0,
            min_url_valid_pct=85,
            min_domain_resolved_pct=75,
            min_email_plausible_pct=60,
            min_lines_per_s=50
        )
        
        # Mock pipeline results
        results = [
            {
                "step": "quality.score",
                "out": {"avg_quality_score": 85.5},
                "duration_s": 1.0
            },
            {
                "step": "quality.dedupe",
                "out": {"total_records": 1000, "duplicates_removed": 15},
                "duration_s": 2.0
            },
            {
                "step": "http.static",
                "out": {"total_urls": 100, "valid_urls": 90},
                "duration_s": 3.0
            },
            {
                "step": "enrich.dns",
                "out": {"total_domains": 50, "resolved_domains": 40},
                "duration_s": 1.5
            },
            {
                "step": "enrich.email",
                "out": {"total_emails": 200, "plausible_emails": 140},
                "duration_s": 2.5
            }
        ]
        
        context = {"total_lines_processed": 500}
        
        kpis = calculator.calculate_final_kpis(context, results)
        
        # Check calculated values
        assert kpis["actual_kpis"]["quality_score"] == 85.5
        assert kpis["actual_kpis"]["dup_pct"] == 1.5  # 15/1000 * 100
        assert kpis["actual_kpis"]["url_valid_pct"] == 90.0  # 90/100 * 100
        assert kpis["actual_kpis"]["domain_resolved_pct"] == 80.0  # 40/50 * 100
        assert kpis["actual_kpis"]["email_plausible_pct"] == 70.0  # 140/200 * 100
        assert kpis["actual_kpis"]["lines_per_s"] == 50.0  # 500/10 total duration
        
        # Check KPI comparisons
        assert kpis["kpi_comparison"]["quality_score_met"] is True  # 85.5 >= 80
        assert kpis["kpi_comparison"]["dup_pct_met"] is True  # 1.5 <= 2.0
        assert kpis["kpi_comparison"]["url_valid_pct_met"] is True  # 90 >= 85
        assert kpis["kpi_comparison"]["domain_resolved_pct_met"] is True  # 80 >= 75
        assert kpis["kpi_comparison"]["email_plausible_pct_met"] is True  # 70 >= 60
        assert kpis["kpi_comparison"]["lines_per_s_met"] is True  # 50 >= 50
        
        assert kpis["all_kpis_met"] is True

    def test_kpi_calculation_failures(self):
        """Test KPI calculation when targets are not met."""
        calculator = KPICalculator(
            min_quality_score=90,  # High requirement
            max_dup_pct=1.0,       # Low tolerance
            min_url_valid_pct=95   # High requirement
        )
        
        results = [
            {
                "step": "quality.score",
                "out": {"avg_quality_score": 75.0},  # Below target
                "duration_s": 1.0
            },
            {
                "step": "quality.dedupe", 
                "out": {"total_records": 100, "duplicates_removed": 3},  # 3% > 1%
                "duration_s": 1.0
            },
            {
                "step": "http.static",
                "out": {"total_urls": 100, "valid_urls": 85},  # 85% < 95%
                "duration_s": 1.0
            }
        ]
        
        context = {"total_lines_processed": 100}
        
        kpis = calculator.calculate_final_kpis(context, results)
        
        # All should fail
        assert kpis["kpi_comparison"]["quality_score_met"] is False
        assert kpis["kpi_comparison"]["dup_pct_met"] is False
        assert kpis["kpi_comparison"]["url_valid_pct_met"] is False
        assert kpis["all_kpis_met"] is False