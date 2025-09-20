#!/usr/bin/env python3
"""
Cost management and monitoring for Apify scrapers.

This module provides cost tracking, budget enforcement, and real-time monitoring
for Apify credit usage across different scrapers.
"""

import time
import logging
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ScraperType(Enum):
    """Types of scrapers with different cost profiles."""
    GOOGLE_PLACES = "google_places"
    GOOGLE_MAPS_CONTACTS = "google_maps_contacts"
    LINKEDIN_PREMIUM = "linkedin_premium"


class CostAlert(NamedTuple):
    """Represents a cost-related alert."""
    timestamp: float
    alert_type: str
    scraper: ScraperType
    message: str
    current_cost: float
    budget_limit: float


@dataclass
class ScraperCostProfile:
    """Cost profile for a specific scraper."""
    scraper_type: ScraperType
    min_cost_per_operation: float
    max_cost_per_operation: float
    avg_cost_per_operation: float
    cost_variance: float = 0.2  # 20% variance by default
    
    # Rate limiting
    max_operations_per_minute: int = 60
    max_operations_per_hour: int = 1000
    
    # Budget controls
    max_daily_cost: float = 500.0
    max_hourly_cost: float = 100.0
    warning_threshold: float = 0.8  # Warn at 80% of budget


# Default cost profiles based on Apify documentation
DEFAULT_COST_PROFILES = {
    ScraperType.GOOGLE_PLACES: ScraperCostProfile(
        scraper_type=ScraperType.GOOGLE_PLACES,
        min_cost_per_operation=1.0,
        max_cost_per_operation=5.0,
        avg_cost_per_operation=3.0,
        max_operations_per_minute=30,
        max_daily_cost=200.0,
        max_hourly_cost=50.0
    ),
    ScraperType.GOOGLE_MAPS_CONTACTS: ScraperCostProfile(
        scraper_type=ScraperType.GOOGLE_MAPS_CONTACTS,
        min_cost_per_operation=1.0,
        max_cost_per_operation=3.0,
        avg_cost_per_operation=2.0,
        max_operations_per_minute=50,
        max_daily_cost=300.0,
        max_hourly_cost=75.0
    ),
    ScraperType.LINKEDIN_PREMIUM: ScraperCostProfile(
        scraper_type=ScraperType.LINKEDIN_PREMIUM,
        min_cost_per_operation=10.0,
        max_cost_per_operation=50.0,
        avg_cost_per_operation=30.0,
        max_operations_per_minute=10,
        max_daily_cost=1000.0,
        max_hourly_cost=200.0
    )
}


@dataclass
class CostTrackingEntry:
    """Single cost tracking entry."""
    timestamp: float
    scraper_type: ScraperType
    operation_id: str
    estimated_cost: float
    actual_cost: Optional[float] = None
    operation_details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class BudgetConfig:
    """Budget configuration for cost management."""
    total_daily_budget: float = 1000.0
    total_hourly_budget: float = 200.0
    
    # Per-scraper budgets
    scraper_budgets: Dict[ScraperType, float] = field(default_factory=lambda: {
        ScraperType.GOOGLE_PLACES: 200.0,
        ScraperType.GOOGLE_MAPS_CONTACTS: 300.0,
        ScraperType.LINKEDIN_PREMIUM: 500.0
    })
    
    # Alert thresholds
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    
    # Cost control settings
    enable_auto_shutdown: bool = True
    enable_cost_optimization: bool = True
    cost_per_result_target: float = 5.0  # Target cost per enriched result


class CostManager:
    """Manages costs and budgets for Apify scrapers."""
    
    def __init__(self, budget_config: Optional[BudgetConfig] = None):
        self.budget_config = budget_config or BudgetConfig()
        self.cost_profiles = DEFAULT_COST_PROFILES.copy()
        
        # Tracking data
        self.cost_entries: List[CostTrackingEntry] = []
        self.alerts: List[CostAlert] = []
        self.session_start_time = time.time()
        
        # Real-time counters
        self.current_costs: Dict[ScraperType, float] = defaultdict(float)
        self.operation_counts: Dict[ScraperType, int] = defaultdict(int)
        self.last_hour_costs: Dict[ScraperType, List[float]] = defaultdict(list)
        
        logger.info(f"CostManager initialized with daily budget: {self.budget_config.total_daily_budget}")
    
    def estimate_operation_cost(
        self, 
        scraper_type: ScraperType, 
        operation_details: Dict[str, Any]
    ) -> float:
        """Estimate the cost of a scraper operation."""
        
        profile = self.cost_profiles.get(scraper_type)
        if not profile:
            logger.warning(f"No cost profile for {scraper_type}, using default")
            return 10.0
        
        # Base cost estimation
        base_cost = profile.avg_cost_per_operation
        
        # Adjust based on operation complexity
        complexity_multiplier = self._calculate_complexity_multiplier(scraper_type, operation_details)
        estimated_cost = base_cost * complexity_multiplier
        
        # Apply variance
        variance = profile.cost_variance
        min_cost = estimated_cost * (1 - variance)
        max_cost = estimated_cost * (1 + variance)
        
        # Return conservative estimate (slightly higher than average)
        return min(max_cost, estimated_cost * 1.1)
    
    def _calculate_complexity_multiplier(
        self, 
        scraper_type: ScraperType, 
        operation_details: Dict[str, Any]
    ) -> float:
        """Calculate complexity multiplier based on operation details."""
        
        multiplier = 1.0
        
        if scraper_type == ScraperType.LINKEDIN_PREMIUM:
            # LinkedIn complexity factors
            max_profiles = operation_details.get('maxProfiles', 5)
            multiplier *= min(2.0, 1.0 + (max_profiles - 5) * 0.1)
            
            # Multiple search terms increase cost
            search_terms = operation_details.get('searchTerms', [])
            if isinstance(search_terms, list):
                multiplier *= min(1.5, 1.0 + len(search_terms) * 0.05)
        
        elif scraper_type == ScraperType.GOOGLE_PLACES:
            # Google Places complexity factors
            max_places = operation_details.get('max_places_per_search', 10)
            multiplier *= min(1.5, 1.0 + (max_places - 10) * 0.02)
            
        elif scraper_type == ScraperType.GOOGLE_MAPS_CONTACTS:
            # Contact enrichment complexity
            max_enrichments = operation_details.get('max_contact_enrichments', 50)
            multiplier *= min(2.0, 1.0 + (max_enrichments - 50) * 0.01)
        
        return max(0.5, min(3.0, multiplier))  # Cap between 0.5x and 3.0x
    
    def pre_operation_check(
        self, 
        scraper_type: ScraperType, 
        operation_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if operation should proceed based on budget constraints."""
        
        estimated_cost = self.estimate_operation_cost(scraper_type, operation_details)
        
        # Check various budget limits
        checks = {
            'can_proceed': True,
            'estimated_cost': estimated_cost,
            'warnings': [],
            'blocks': [],
            'recommended_adjustments': {}
        }
        
        # Check total daily budget
        current_daily_total = sum(self.current_costs.values())
        if current_daily_total + estimated_cost > self.budget_config.total_daily_budget:
            checks['can_proceed'] = False
            checks['blocks'].append(f"Would exceed daily budget: {current_daily_total + estimated_cost:.1f} > {self.budget_config.total_daily_budget}")
        
        # Check scraper-specific budget
        scraper_budget = self.budget_config.scraper_budgets.get(scraper_type, float('inf'))
        current_scraper_cost = self.current_costs[scraper_type]
        if current_scraper_cost + estimated_cost > scraper_budget:
            checks['can_proceed'] = False
            checks['blocks'].append(f"Would exceed {scraper_type.value} budget: {current_scraper_cost + estimated_cost:.1f} > {scraper_budget}")
        
        # Check hourly rate limits
        hourly_cost = self._get_hourly_cost(scraper_type)
        profile = self.cost_profiles.get(scraper_type)
        if profile and hourly_cost + estimated_cost > profile.max_hourly_cost:
            checks['warnings'].append(f"Approaching hourly cost limit for {scraper_type.value}")
        
        # Warning thresholds
        daily_usage_pct = (current_daily_total + estimated_cost) / self.budget_config.total_daily_budget
        if daily_usage_pct > self.budget_config.warning_threshold:
            checks['warnings'].append(f"High daily budget usage: {daily_usage_pct:.1%}")
        
        # Suggest cost optimizations
        if estimated_cost > profile.avg_cost_per_operation * 1.5:
            checks['recommended_adjustments'] = self._suggest_cost_optimizations(scraper_type, operation_details)
        
        return checks
    
    def track_operation_start(
        self, 
        scraper_type: ScraperType, 
        operation_id: str,
        operation_details: Dict[str, Any]
    ) -> CostTrackingEntry:
        """Track the start of an operation."""
        
        estimated_cost = self.estimate_operation_cost(scraper_type, operation_details)
        
        entry = CostTrackingEntry(
            timestamp=time.time(),
            scraper_type=scraper_type,
            operation_id=operation_id,
            estimated_cost=estimated_cost,
            operation_details=operation_details
        )
        
        self.cost_entries.append(entry)
        self.current_costs[scraper_type] += estimated_cost
        self.operation_counts[scraper_type] += 1
        
        logger.info(
            f"Started {scraper_type.value} operation {operation_id}, "
            f"estimated cost: {estimated_cost:.1f} credits"
        )
        
        return entry
    
    def track_operation_complete(
        self, 
        operation_id: str, 
        actual_cost: Optional[float] = None,
        success: bool = True,
        results_count: int = 0
    ) -> None:
        """Track the completion of an operation."""
        
        # Find the corresponding entry
        entry = None
        for e in self.cost_entries:
            if e.operation_id == operation_id:
                entry = e
                break
        
        if not entry:
            logger.warning(f"No tracking entry found for operation {operation_id}")
            return
        
        # Update entry
        entry.actual_cost = actual_cost
        entry.success = success
        
        # Adjust current costs if actual cost is different
        if actual_cost is not None:
            cost_diff = actual_cost - entry.estimated_cost
            self.current_costs[entry.scraper_type] += cost_diff
        
        # Track cost per result efficiency
        if success and results_count > 0:
            final_cost = actual_cost or entry.estimated_cost
            cost_per_result = final_cost / results_count
            
            if cost_per_result > self.budget_config.cost_per_result_target:
                self._generate_alert(
                    "HIGH_COST_PER_RESULT",
                    entry.scraper_type,
                    f"High cost per result: {cost_per_result:.1f} > {self.budget_config.cost_per_result_target}",
                    final_cost,
                    self.budget_config.cost_per_result_target * results_count
                )
        
        logger.info(
            f"Completed {entry.scraper_type.value} operation {operation_id}, "
            f"success: {success}, results: {results_count}"
        )
    
    def get_real_time_costs(self) -> Dict[str, Any]:
        """Get real-time cost information."""
        
        current_time = time.time()
        session_duration_hours = (current_time - self.session_start_time) / 3600
        
        # Calculate hourly costs
        hourly_costs = {}
        for scraper_type in ScraperType:
            hourly_costs[scraper_type.value] = self._get_hourly_cost(scraper_type)
        
        # Calculate total costs
        total_current_cost = sum(self.current_costs.values())
        total_daily_budget = self.budget_config.total_daily_budget
        
        # Calculate efficiency metrics
        total_operations = sum(self.operation_counts.values())
        avg_cost_per_operation = total_current_cost / max(total_operations, 1)
        
        return {
            'timestamp': current_time,
            'session_duration_hours': round(session_duration_hours, 2),
            'current_costs': {k.value: v for k, v in self.current_costs.items()},
            'total_current_cost': total_current_cost,
            'daily_budget_usage_pct': (total_current_cost / total_daily_budget) * 100,
            'hourly_costs': hourly_costs,
            'operation_counts': {k.value: v for k, v in self.operation_counts.items()},
            'avg_cost_per_operation': avg_cost_per_operation,
            'remaining_budget': total_daily_budget - total_current_cost,
            'alerts_count': len(self.alerts),
            'cost_efficiency_score': self._calculate_efficiency_score()
        }
    
    def _get_hourly_cost(self, scraper_type: ScraperType) -> float:
        """Calculate cost for the last hour for a scraper."""
        
        current_time = time.time()
        hour_ago = current_time - 3600
        
        hourly_cost = 0.0
        for entry in self.cost_entries:
            if (entry.scraper_type == scraper_type and 
                entry.timestamp >= hour_ago):
                hourly_cost += entry.actual_cost or entry.estimated_cost
        
        return hourly_cost
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall cost efficiency score (0-100)."""
        
        if not self.cost_entries:
            return 100.0
        
        # Factors that contribute to efficiency score
        factors = []
        
        # Budget utilization efficiency (prefer 70-90% usage)
        total_cost = sum(self.current_costs.values())
        budget_usage = total_cost / self.budget_config.total_daily_budget
        
        if 0.7 <= budget_usage <= 0.9:
            budget_score = 100.0
        elif budget_usage < 0.7:
            budget_score = 80.0 + (budget_usage / 0.7) * 20.0
        else:
            budget_score = max(0.0, 100.0 - (budget_usage - 0.9) * 200.0)
        
        factors.append(budget_score)
        
        # Success rate efficiency
        successful_ops = sum(1 for e in self.cost_entries if e.success)
        success_rate = successful_ops / len(self.cost_entries)
        factors.append(success_rate * 100)
        
        # Cost prediction accuracy
        actual_costs = [e.actual_cost for e in self.cost_entries if e.actual_cost is not None]
        if actual_costs:
            estimated_costs = [e.estimated_cost for e in self.cost_entries if e.actual_cost is not None]
            accuracy_scores = []
            for actual, estimated in zip(actual_costs, estimated_costs):
                accuracy = 100.0 - min(100.0, abs(actual - estimated) / estimated * 100)
                accuracy_scores.append(accuracy)
            factors.append(sum(accuracy_scores) / len(accuracy_scores))
        
        return sum(factors) / len(factors)
    
    def _suggest_cost_optimizations(
        self, 
        scraper_type: ScraperType, 
        operation_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest optimizations to reduce operation costs."""
        
        suggestions = {}
        
        if scraper_type == ScraperType.LINKEDIN_PREMIUM:
            max_profiles = operation_details.get('maxProfiles', 5)
            if max_profiles > 5:
                suggestions['maxProfiles'] = min(5, max_profiles)
                suggestions['reason'] = "Reduce maxProfiles to decrease cost"
            
            search_terms = operation_details.get('searchTerms', [])
            if isinstance(search_terms, list) and len(search_terms) > 3:
                suggestions['searchTerms'] = search_terms[:3]
                suggestions['reason'] = "Limit search terms to reduce cost"
        
        elif scraper_type == ScraperType.GOOGLE_PLACES:
            max_places = operation_details.get('max_places_per_search', 10)
            if max_places > 10:
                suggestions['max_places_per_search'] = 10
                suggestions['reason'] = "Reduce max places to decrease cost"
        
        elif scraper_type == ScraperType.GOOGLE_MAPS_CONTACTS:
            max_enrichments = operation_details.get('max_contact_enrichments', 50)
            if max_enrichments > 25:
                suggestions['max_contact_enrichments'] = 25
                suggestions['reason'] = "Reduce enrichments to decrease cost"
        
        return suggestions
    
    def _generate_alert(
        self, 
        alert_type: str, 
        scraper: ScraperType, 
        message: str,
        current_cost: float, 
        budget_limit: float
    ) -> None:
        """Generate a cost alert."""
        
        alert = CostAlert(
            timestamp=time.time(),
            alert_type=alert_type,
            scraper=scraper,
            message=message,
            current_cost=current_cost,
            budget_limit=budget_limit
        )
        
        self.alerts.append(alert)
        logger.warning(f"Cost Alert [{alert_type}]: {message}")
    
    def check_budget_thresholds(self) -> List[CostAlert]:
        """Check budget thresholds and generate alerts if needed."""
        
        new_alerts = []
        current_time = time.time()
        
        # Check total daily budget
        total_cost = sum(self.current_costs.values())
        daily_usage_pct = total_cost / self.budget_config.total_daily_budget
        
        if daily_usage_pct >= self.budget_config.critical_threshold:
            alert = CostAlert(
                timestamp=current_time,
                alert_type="CRITICAL_BUDGET",
                scraper=ScraperType.GOOGLE_PLACES,  # Generic
                message=f"Critical budget usage: {daily_usage_pct:.1%}",
                current_cost=total_cost,
                budget_limit=self.budget_config.total_daily_budget
            )
            new_alerts.append(alert)
            
        elif daily_usage_pct >= self.budget_config.warning_threshold:
            alert = CostAlert(
                timestamp=current_time,
                alert_type="WARNING_BUDGET",
                scraper=ScraperType.GOOGLE_PLACES,  # Generic
                message=f"High budget usage: {daily_usage_pct:.1%}",
                current_cost=total_cost,
                budget_limit=self.budget_config.total_daily_budget
            )
            new_alerts.append(alert)
        
        # Check per-scraper budgets
        for scraper_type, budget in self.budget_config.scraper_budgets.items():
            scraper_cost = self.current_costs[scraper_type]
            usage_pct = scraper_cost / budget
            
            if usage_pct >= self.budget_config.critical_threshold:
                alert = CostAlert(
                    timestamp=current_time,
                    alert_type="CRITICAL_SCRAPER_BUDGET",
                    scraper=scraper_type,
                    message=f"Critical {scraper_type.value} budget usage: {usage_pct:.1%}",
                    current_cost=scraper_cost,
                    budget_limit=budget
                )
                new_alerts.append(alert)
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary."""
        
        current_costs = self.get_real_time_costs()
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]  # Last hour
        
        # Calculate cost trends
        cost_trends = {}
        for scraper_type in ScraperType:
            recent_entries = [
                e for e in self.cost_entries 
                if e.scraper_type == scraper_type and time.time() - e.timestamp < 3600
            ]
            if recent_entries:
                avg_recent_cost = sum(e.actual_cost or e.estimated_cost for e in recent_entries) / len(recent_entries)
                cost_trends[scraper_type.value] = {
                    'avg_cost_last_hour': avg_recent_cost,
                    'operations_last_hour': len(recent_entries),
                    'trend': 'stable'  # Could be enhanced with trend analysis
                }
        
        return {
            'current_status': current_costs,
            'recent_alerts': [
                {
                    'type': a.alert_type,
                    'scraper': a.scraper.value,
                    'message': a.message,
                    'timestamp': a.timestamp
                } for a in recent_alerts
            ],
            'cost_trends': cost_trends,
            'efficiency_recommendations': self._get_efficiency_recommendations(),
            'budget_projections': self._calculate_budget_projections()
        }
    
    def _get_efficiency_recommendations(self) -> List[str]:
        """Get recommendations for improving cost efficiency."""
        
        recommendations = []
        
        # Analyze cost per result
        if self.cost_entries:
            total_cost = sum(e.actual_cost or e.estimated_cost for e in self.cost_entries)
            successful_ops = sum(1 for e in self.cost_entries if e.success)
            
            if successful_ops > 0:
                avg_cost_per_success = total_cost / successful_ops
                if avg_cost_per_success > self.budget_config.cost_per_result_target * 2:
                    recommendations.append("Consider reducing batch sizes to improve cost efficiency")
        
        # Check for high-cost operations
        high_cost_entries = [e for e in self.cost_entries if (e.actual_cost or e.estimated_cost) > 50]
        if len(high_cost_entries) > len(self.cost_entries) * 0.2:
            recommendations.append("High proportion of expensive operations - review operation parameters")
        
        # Budget utilization
        total_cost = sum(self.current_costs.values())
        if total_cost < self.budget_config.total_daily_budget * 0.5:
            recommendations.append("Budget underutilized - consider increasing operation scope")
        
        return recommendations
    
    def _calculate_budget_projections(self) -> Dict[str, Any]:
        """Calculate budget projections based on current usage."""
        
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        if session_duration < 300:  # Less than 5 minutes
            return {"projection": "insufficient_data"}
        
        total_cost = sum(self.current_costs.values())
        cost_rate_per_hour = total_cost / (session_duration / 3600)
        
        # Project to end of day (assuming 8-hour work day)
        hours_remaining = max(0, 8 - (session_duration / 3600))
        projected_daily_cost = total_cost + (cost_rate_per_hour * hours_remaining)
        
        return {
            "current_cost_rate_per_hour": round(cost_rate_per_hour, 2),
            "projected_daily_cost": round(projected_daily_cost, 2),
            "budget_utilization_projection": round(projected_daily_cost / self.budget_config.total_daily_budget * 100, 1),
            "hours_remaining_at_current_rate": round(
                (self.budget_config.total_daily_budget - total_cost) / max(cost_rate_per_hour, 1), 2
            )
        }


def create_cost_manager(job_config: Dict[str, Any]) -> CostManager:
    """Create a cost manager from job configuration."""
    
    apify_config = job_config.get("apify", {})
    
    # Extract budget configuration
    budget_config = BudgetConfig()
    
    if "budget" in apify_config:
        budget_settings = apify_config["budget"]
        budget_config.total_daily_budget = budget_settings.get("total_daily_budget", 1000.0)
        budget_config.total_hourly_budget = budget_settings.get("total_hourly_budget", 200.0)
        budget_config.warning_threshold = budget_settings.get("warning_threshold", 0.8)
        budget_config.critical_threshold = budget_settings.get("critical_threshold", 0.95)
        budget_config.enable_auto_shutdown = budget_settings.get("enable_auto_shutdown", True)
        budget_config.cost_per_result_target = budget_settings.get("cost_per_result_target", 5.0)
    
    return CostManager(budget_config)