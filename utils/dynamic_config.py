#!/usr/bin/env python3
"""
Dynamic configuration management for Apify scrapers.

This module provides budget-based configuration adjustment and adaptive
scraper selection based on real-time conditions and constraints.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from .cost_manager import ScraperType, CostManager

logger = logging.getLogger(__name__)


class BudgetMode(Enum):
    """Budget modes for dynamic configuration."""
    MINIMAL = "minimal"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    UNLIMITED = "unlimited"


class ConfigPriority(Enum):
    """Configuration priorities for resource allocation."""
    CONTACTS = "contacts"
    EXECUTIVES = "executives"
    PLACES = "places"
    COVERAGE = "coverage"
    QUALITY = "quality"
    SPEED = "speed"


@dataclass
class ConfigurationProfile:
    """Configuration profile for different budget modes."""
    budget_mode: BudgetMode
    max_addresses: int
    google_places_config: Dict[str, Any]
    google_maps_contacts_config: Dict[str, Any]
    linkedin_premium_config: Dict[str, Any]
    retry_config: Dict[str, Any]
    timeout_config: Dict[str, Any]
    quality_thresholds: Dict[str, Any]
    
    estimated_cost_range: Tuple[float, float] = (0.0, 0.0)
    recommended_batch_size: int = 10
    description: str = ""


# Pre-defined configuration profiles
DEFAULT_PROFILES = {
    BudgetMode.MINIMAL: ConfigurationProfile(
        budget_mode=BudgetMode.MINIMAL,
        max_addresses=5,
        google_places_config={
            "enabled": True,
            "max_places_per_search": 5,
            "timeout_seconds": 180
        },
        google_maps_contacts_config={
            "enabled": False,
            "max_contact_enrichments": 0,
            "timeout_seconds": 120
        },
        linkedin_premium_config={
            "enabled": False,
            "max_linkedin_searches": 0,
            "max_profiles_per_company": 0,
            "timeout_seconds": 300
        },
        retry_config={
            "max_retries": 1,
            "max_cost_per_search": 25.0,
            "strategies": ["simplified_name"]
        },
        timeout_config={
            "base_timeout": 180,
            "escalation_factor": 1.0
        },
        quality_thresholds={
            "min_confidence": 0.6,
            "require_contact_info": False
        },
        estimated_cost_range=(15.0, 50.0),
        recommended_batch_size=5,
        description="Minimal cost configuration for testing and development"
    ),
    
    BudgetMode.CONSERVATIVE: ConfigurationProfile(
        budget_mode=BudgetMode.CONSERVATIVE,
        max_addresses=15,
        google_places_config={
            "enabled": True,
            "max_places_per_search": 8,
            "timeout_seconds": 240
        },
        google_maps_contacts_config={
            "enabled": True,
            "max_contact_enrichments": 10,
            "timeout_seconds": 180
        },
        linkedin_premium_config={
            "enabled": False,
            "max_linkedin_searches": 0,
            "max_profiles_per_company": 0,
            "timeout_seconds": 300
        },
        retry_config={
            "max_retries": 2,
            "max_cost_per_search": 50.0,
            "strategies": ["simplified_name", "broader_search"]
        },
        timeout_config={
            "base_timeout": 240,
            "escalation_factor": 1.2
        },
        quality_thresholds={
            "min_confidence": 0.7,
            "require_contact_info": False
        },
        estimated_cost_range=(75.0, 200.0),
        recommended_batch_size=10,
        description="Conservative configuration balancing cost and basic enrichment"
    ),
    
    BudgetMode.BALANCED: ConfigurationProfile(
        budget_mode=BudgetMode.BALANCED,
        max_addresses=25,
        google_places_config={
            "enabled": True,
            "max_places_per_search": 10,
            "timeout_seconds": 300
        },
        google_maps_contacts_config={
            "enabled": True,
            "max_contact_enrichments": 25,
            "timeout_seconds": 240
        },
        linkedin_premium_config={
            "enabled": True,
            "max_linkedin_searches": 10,
            "max_profiles_per_company": 3,
            "timeout_seconds": 450
        },
        retry_config={
            "max_retries": 3,
            "max_cost_per_search": 100.0,
            "strategies": ["simplified_name", "alternative_positions", "broader_search"]
        },
        timeout_config={
            "base_timeout": 300,
            "escalation_factor": 1.3
        },
        quality_thresholds={
            "min_confidence": 0.75,
            "require_contact_info": True
        },
        estimated_cost_range=(200.0, 500.0),
        recommended_batch_size=15,
        description="Balanced configuration for production use with moderate enrichment"
    ),
    
    BudgetMode.AGGRESSIVE: ConfigurationProfile(
        budget_mode=BudgetMode.AGGRESSIVE,
        max_addresses=50,
        google_places_config={
            "enabled": True,
            "max_places_per_search": 15,
            "timeout_seconds": 450
        },
        google_maps_contacts_config={
            "enabled": True,
            "max_contact_enrichments": 50,
            "timeout_seconds": 360
        },
        linkedin_premium_config={
            "enabled": True,
            "max_linkedin_searches": 25,
            "max_profiles_per_company": 5,
            "timeout_seconds": 600
        },
        retry_config={
            "max_retries": 3,
            "max_cost_per_search": 200.0,
            "strategies": ["simplified_name", "alternative_positions", "broader_search", "increased_timeout"]
        },
        timeout_config={
            "base_timeout": 450,
            "escalation_factor": 1.5
        },
        quality_thresholds={
            "min_confidence": 0.8,
            "require_contact_info": True
        },
        estimated_cost_range=(500.0, 1200.0),
        recommended_batch_size=25,
        description="Aggressive configuration for comprehensive enrichment"
    ),
    
    BudgetMode.UNLIMITED: ConfigurationProfile(
        budget_mode=BudgetMode.UNLIMITED,
        max_addresses=100,
        google_places_config={
            "enabled": True,
            "max_places_per_search": 20,
            "timeout_seconds": 600
        },
        google_maps_contacts_config={
            "enabled": True,
            "max_contact_enrichments": 100,
            "timeout_seconds": 480
        },
        linkedin_premium_config={
            "enabled": True,
            "max_linkedin_searches": 50,
            "max_profiles_per_company": 8,
            "timeout_seconds": 900
        },
        retry_config={
            "max_retries": 5,
            "max_cost_per_search": 500.0,
            "strategies": ["simplified_name", "alternative_positions", "broader_search", "increased_timeout", "reduced_batch"]
        },
        timeout_config={
            "base_timeout": 600,
            "escalation_factor": 2.0
        },
        quality_thresholds={
            "min_confidence": 0.85,
            "require_contact_info": True
        },
        estimated_cost_range=(1200.0, 3000.0),
        recommended_batch_size=50,
        description="Unlimited configuration for maximum enrichment quality"
    )
}


class DynamicConfigurationManager:
    """Manages dynamic configuration based on budget and priorities."""
    
    def __init__(self, cost_manager: Optional[CostManager] = None):
        self.cost_manager = cost_manager
        self.profiles = DEFAULT_PROFILES.copy()
        self.current_profile: Optional[ConfigurationProfile] = None
        self.configuration_history: List[Dict[str, Any]] = []
        
        logger.info("DynamicConfigurationManager initialized")
    
    def determine_optimal_configuration(
        self,
        base_config: Dict[str, Any],
        available_budget: float,
        priorities: List[ConfigPriority],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal configuration based on budget and priorities."""
        
        # Select base profile based on budget
        budget_mode = self._select_budget_mode(available_budget, context)
        base_profile = self.profiles[budget_mode]
        
        logger.info(f"Selected {budget_mode.value} configuration profile (budget: {available_budget:.1f})")
        
        # Customize profile based on priorities
        customized_profile = self._customize_for_priorities(base_profile, priorities, available_budget)
        
        # Apply context-specific adjustments
        final_config = self._apply_context_adjustments(customized_profile, context, base_config)
        
        # Store configuration decision
        self._record_configuration_decision(
            budget_mode, customized_profile, priorities, context, available_budget
        )
        
        self.current_profile = customized_profile
        return final_config
    
    def _select_budget_mode(self, available_budget: float, context: Dict[str, Any]) -> BudgetMode:
        """Select appropriate budget mode based on available budget."""
        
        # Budget thresholds for mode selection
        thresholds = {
            BudgetMode.MINIMAL: 50.0,
            BudgetMode.CONSERVATIVE: 200.0,
            BudgetMode.BALANCED: 500.0,
            BudgetMode.AGGRESSIVE: 1200.0,
            BudgetMode.UNLIMITED: float('inf')
        }
        
        # Consider dry run mode
        if context.get("dry_run", False):
            return BudgetMode.MINIMAL
        
        # Consider time constraints
        time_budget_min = context.get("time_budget_min", 0)
        if time_budget_min > 0 and time_budget_min < 30:
            # Short time budget suggests faster, less comprehensive mode
            if available_budget > 500:
                return BudgetMode.BALANCED
            else:
                return BudgetMode.CONSERVATIVE
        
        # Select based on budget
        for mode, threshold in thresholds.items():
            if available_budget <= threshold:
                return mode
        
        return BudgetMode.UNLIMITED
    
    def _customize_for_priorities(
        self,
        base_profile: ConfigurationProfile,
        priorities: List[ConfigPriority],
        available_budget: float
    ) -> ConfigurationProfile:
        """Customize configuration profile based on priorities."""
        
        profile = deepcopy(base_profile)
        
        # Adjust based on priority weights
        priority_weights = {p: 1.0 for p in priorities}
        
        if ConfigPriority.CONTACTS in priorities:
            # Boost contact enrichment
            profile.google_maps_contacts_config["enabled"] = True
            if available_budget > 100:
                profile.google_maps_contacts_config["max_contact_enrichments"] = min(
                    profile.google_maps_contacts_config["max_contact_enrichments"] * 2,
                    int(available_budget / 3)  # Roughly 3 credits per contact
                )
        
        if ConfigPriority.EXECUTIVES in priorities:
            # Boost LinkedIn premium
            if available_budget > 200:
                profile.linkedin_premium_config["enabled"] = True
                profile.linkedin_premium_config["max_linkedin_searches"] = min(
                    profile.linkedin_premium_config["max_linkedin_searches"] * 2,
                    int(available_budget / 30)  # Roughly 30 credits per search
                )
                profile.linkedin_premium_config["max_profiles_per_company"] += 2
        
        if ConfigPriority.COVERAGE in priorities:
            # Increase overall coverage
            profile.max_addresses = min(
                int(profile.max_addresses * 1.5),
                int(available_budget / 10)  # Roughly 10 credits per address minimum
            )
            profile.google_places_config["max_places_per_search"] += 5
        
        if ConfigPriority.QUALITY in priorities:
            # Increase quality thresholds and retry attempts
            profile.quality_thresholds["min_confidence"] = min(0.9, profile.quality_thresholds["min_confidence"] + 0.1)
            profile.retry_config["max_retries"] += 1
            profile.retry_config["max_cost_per_search"] *= 1.5
        
        if ConfigPriority.SPEED in priorities:
            # Reduce timeouts and retries for faster execution
            profile.timeout_config["base_timeout"] = int(profile.timeout_config["base_timeout"] * 0.8)
            profile.retry_config["max_retries"] = max(1, profile.retry_config["max_retries"] - 1)
            
            # Reduce batch sizes for parallelization
            profile.recommended_batch_size = max(5, int(profile.recommended_batch_size * 0.7))
        
        if ConfigPriority.PLACES in priorities:
            # Focus on place data quality
            profile.google_places_config["max_places_per_search"] += 5
            profile.google_places_config["timeout_seconds"] += 60
        
        return profile
    
    def _apply_context_adjustments(
        self,
        profile: ConfigurationProfile,
        context: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply context-specific adjustments to the configuration."""
        
        # Start with base configuration
        final_config = deepcopy(base_config)
        
        # Apply profile settings to apify configuration
        if "apify" not in final_config:
            final_config["apify"] = {}
        
        apify_config = final_config["apify"]
        
        # Core settings
        apify_config["max_addresses"] = profile.max_addresses
        
        # Google Places configuration
        apify_config["google_places"] = profile.google_places_config.copy()
        
        # Google Maps contacts configuration
        apify_config["google_maps_contacts"] = profile.google_maps_contacts_config.copy()
        
        # LinkedIn Premium configuration
        apify_config["linkedin_premium"] = profile.linkedin_premium_config.copy()
        
        # Retry configuration
        apify_config["retry_settings"] = profile.retry_config.copy()
        
        # Timeout configuration
        for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
            if scraper in apify_config:
                apify_config[scraper]["timeout_seconds"] = profile.timeout_config["base_timeout"]
        
        # Quality control settings
        if "quality_control" not in apify_config:
            apify_config["quality_control"] = {}
        apify_config["quality_control"].update(profile.quality_thresholds)
        
        # Context-specific adjustments
        
        # Adjust for data size
        total_addresses = context.get("total_addresses", 0)
        if total_addresses > 0:
            # Scale batch size based on total data size
            optimal_batches = max(1, total_addresses // 50)  # Aim for 50 addresses per batch
            if optimal_batches > 1:
                apify_config["max_addresses"] = min(
                    apify_config["max_addresses"],
                    total_addresses // optimal_batches
                )
        
        # Adjust for time constraints
        time_budget_min = context.get("time_budget_min", 0)
        if time_budget_min > 0 and time_budget_min < 60:
            # Tight time budget - reduce timeouts and retries
            for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
                if scraper in apify_config:
                    apify_config[scraper]["timeout_seconds"] = int(
                        apify_config[scraper]["timeout_seconds"] * 0.7
                    )
            apify_config["retry_settings"]["max_retries"] = max(
                1, apify_config["retry_settings"]["max_retries"] - 1
            )
        
        # Adjust for worker count
        workers = context.get("workers", 4)
        if workers > 4:
            # More workers available - can handle larger batches
            apify_config["max_addresses"] = min(
                int(apify_config["max_addresses"] * 1.2),
                workers * 5
            )
        
        return final_config
    
    def adjust_configuration_runtime(
        self,
        current_config: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        cost_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust configuration during runtime based on performance and cost metrics."""
        
        if not self.current_profile:
            return current_config
        
        adjusted_config = deepcopy(current_config)
        apify_config = adjusted_config.get("apify", {})
        
        # Analyze performance metrics
        avg_operation_time = performance_metrics.get("avg_operation_time_seconds", 0)
        success_rate = performance_metrics.get("success_rate", 1.0)
        error_rate = performance_metrics.get("error_rate", 0.0)
        
        # Analyze cost metrics
        cost_per_result = cost_metrics.get("cost_per_result", 0)
        budget_burn_rate = cost_metrics.get("budget_burn_rate_pct", 0)
        remaining_budget = cost_metrics.get("remaining_budget", float('inf'))
        
        adjustments_made = []
        
        # Adjust based on success rate
        if success_rate < 0.7:
            # Low success rate - increase timeouts and retries
            for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
                if scraper in apify_config:
                    current_timeout = apify_config[scraper].get("timeout_seconds", 300)
                    apify_config[scraper]["timeout_seconds"] = min(int(current_timeout * 1.3), 900)
            
            retry_settings = apify_config.get("retry_settings", {})
            retry_settings["max_retries"] = min(retry_settings.get("max_retries", 3) + 1, 5)
            adjustments_made.append("Increased timeouts and retries due to low success rate")
        
        # Adjust based on cost efficiency
        if cost_per_result > 10.0 and remaining_budget < 200:
            # High cost per result with low remaining budget - reduce scope
            apify_config["max_addresses"] = max(5, int(apify_config.get("max_addresses", 10) * 0.7))
            
            # Disable expensive operations
            if apify_config.get("linkedin_premium", {}).get("enabled", False):
                apify_config["linkedin_premium"]["enabled"] = False
                adjustments_made.append("Disabled LinkedIn Premium due to high cost per result")
            
            # Reduce contact enrichments
            google_maps = apify_config.get("google_maps_contacts", {})
            if google_maps.get("enabled", False):
                current_enrichments = google_maps.get("max_contact_enrichments", 25)
                google_maps["max_contact_enrichments"] = max(5, int(current_enrichments * 0.5))
                adjustments_made.append("Reduced contact enrichments due to high cost")
        
        # Adjust based on time performance
        if avg_operation_time > 300:  # Operations taking more than 5 minutes
            # Reduce timeouts to prevent hanging operations
            for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
                if scraper in apify_config:
                    current_timeout = apify_config[scraper].get("timeout_seconds", 300)
                    apify_config[scraper]["timeout_seconds"] = max(int(current_timeout * 0.8), 120)
            adjustments_made.append("Reduced timeouts due to slow operation performance")
        
        # Adjust based on budget burn rate
        if budget_burn_rate > 80:  # Burning budget too fast
            # Reduce batch sizes and disable expensive features
            apify_config["max_addresses"] = max(5, int(apify_config.get("max_addresses", 10) * 0.6))
            
            # Reduce retry attempts
            retry_settings = apify_config.get("retry_settings", {})
            retry_settings["max_retries"] = max(1, retry_settings.get("max_retries", 3) - 1)
            retry_settings["max_cost_per_search"] *= 0.7
            
            adjustments_made.append("Reduced scope due to high budget burn rate")
        
        if adjustments_made:
            logger.info(f"Runtime configuration adjustments: {'; '.join(adjustments_made)}")
            
            # Record the adjustment
            self.configuration_history.append({
                "timestamp": time.time(),
                "type": "runtime_adjustment",
                "adjustments": adjustments_made,
                "performance_metrics": performance_metrics,
                "cost_metrics": cost_metrics
            })
        
        return adjusted_config
    
    def _record_configuration_decision(
        self,
        budget_mode: BudgetMode,
        profile: ConfigurationProfile,
        priorities: List[ConfigPriority],
        context: Dict[str, Any],
        available_budget: float
    ) -> None:
        """Record configuration decision for analysis."""
        
        decision_record = {
            "timestamp": time.time(),
            "budget_mode": budget_mode.value,
            "available_budget": available_budget,
            "priorities": [p.value for p in priorities],
            "context_summary": {
                "dry_run": context.get("dry_run", False),
                "total_addresses": context.get("total_addresses", 0),
                "time_budget_min": context.get("time_budget_min", 0),
                "workers": context.get("workers", 4)
            },
            "profile_summary": {
                "max_addresses": profile.max_addresses,
                "estimated_cost_range": profile.estimated_cost_range,
                "google_places_enabled": profile.google_places_config["enabled"],
                "google_maps_enabled": profile.google_maps_contacts_config["enabled"],
                "linkedin_enabled": profile.linkedin_premium_config["enabled"]
            }
        }
        
        self.configuration_history.append(decision_record)
        logger.debug(f"Recorded configuration decision: {budget_mode.value}")
    
    def get_configuration_recommendations(
        self,
        current_config: Dict[str, Any],
        available_budget: float,
        performance_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get recommendations for configuration improvements."""
        
        recommendations = {
            "budget_optimization": [],
            "performance_optimization": [],
            "cost_efficiency": [],
            "alternative_profiles": []
        }
        
        # Analyze current configuration efficiency
        if performance_history:
            avg_cost_per_result = sum(
                h.get("cost_per_result", 0) for h in performance_history
            ) / len(performance_history)
            avg_success_rate = sum(
                h.get("success_rate", 0) for h in performance_history
            ) / len(performance_history)
            
            # Budget optimization recommendations
            if avg_cost_per_result > 8.0:
                recommendations["budget_optimization"].append(
                    "Consider reducing batch sizes or disabling expensive features to improve cost efficiency"
                )
            
            if available_budget > 1000 and avg_cost_per_result < 3.0:
                recommendations["budget_optimization"].append(
                    "Budget allows for more aggressive configuration to increase data quality"
                )
            
            # Performance optimization recommendations
            if avg_success_rate < 0.8:
                recommendations["performance_optimization"].append(
                    "Consider increasing timeouts and retry attempts to improve success rate"
                )
            
            # Cost efficiency recommendations
            efficiency_score = avg_success_rate / max(avg_cost_per_result, 1)
            if efficiency_score < 0.1:
                recommendations["cost_efficiency"].append(
                    "Current configuration has low cost efficiency - consider switching to conservative mode"
                )
        
        # Suggest alternative profiles
        current_mode = self._detect_current_mode(current_config)
        for mode, profile in self.profiles.items():
            if mode != current_mode and available_budget >= profile.estimated_cost_range[0]:
                cost_diff = profile.estimated_cost_range[1] - self.profiles[current_mode].estimated_cost_range[1]
                recommendations["alternative_profiles"].append({
                    "mode": mode.value,
                    "description": profile.description,
                    "cost_difference": cost_diff,
                    "suitability_score": self._calculate_profile_suitability(profile, available_budget)
                })
        
        return recommendations
    
    def _detect_current_mode(self, config: Dict[str, Any]) -> BudgetMode:
        """Detect the current budget mode from configuration."""
        
        apify_config = config.get("apify", {})
        max_addresses = apify_config.get("max_addresses", 10)
        
        linkedin_enabled = apify_config.get("linkedin_premium", {}).get("enabled", False)
        contacts_enabled = apify_config.get("google_maps_contacts", {}).get("enabled", False)
        
        if max_addresses <= 5 and not linkedin_enabled and not contacts_enabled:
            return BudgetMode.MINIMAL
        elif max_addresses <= 15 and not linkedin_enabled:
            return BudgetMode.CONSERVATIVE
        elif max_addresses <= 25 and linkedin_enabled:
            return BudgetMode.BALANCED
        elif max_addresses <= 50:
            return BudgetMode.AGGRESSIVE
        else:
            return BudgetMode.UNLIMITED
    
    def _calculate_profile_suitability(self, profile: ConfigurationProfile, available_budget: float) -> float:
        """Calculate suitability score for a profile given the available budget."""
        
        # Base suitability on budget fit
        min_cost, max_cost = profile.estimated_cost_range
        
        if available_budget < min_cost:
            return 0.0  # Not affordable
        elif available_budget >= max_cost:
            return 1.0  # Perfect fit
        else:
            # Partial fit
            return (available_budget - min_cost) / (max_cost - min_cost)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of configuration management status."""
        
        return {
            "current_profile": self.current_profile.budget_mode.value if self.current_profile else None,
            "configuration_changes": len(self.configuration_history),
            "recent_decisions": self.configuration_history[-5:] if self.configuration_history else [],
            "available_profiles": list(self.profiles.keys()),
            "profile_descriptions": {
                mode.value: profile.description 
                for mode, profile in self.profiles.items()
            }
        }


def create_dynamic_config_manager(cost_manager: Optional[CostManager] = None) -> DynamicConfigurationManager:
    """Create a dynamic configuration manager."""
    return DynamicConfigurationManager(cost_manager)