#!/usr/bin/env python3
"""Tests for the intelligent automation features."""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, List

from utils.retry_manager import LinkedInRetryManager, GoogleMapsRetryManager, RetryConfig, RetryStrategy
from utils.cost_manager import CostManager, ScraperType, BudgetConfig, create_cost_manager
from utils.dynamic_config import DynamicConfigurationManager, ConfigPriority, BudgetMode, create_dynamic_config_manager
from utils.industry_optimizer import IndustryOptimizer, IndustryCategory, OptimizationStrategy, create_industry_optimizer


class TestRetryManager:
    """Test the retry management functionality."""
    
    def test_linkedin_retry_manager_initialization(self):
        """Test LinkedIn retry manager initialization."""
        config = RetryConfig(max_retries=5, max_cost_per_search=200.0)
        manager = LinkedInRetryManager(config)
        
        assert manager.config.max_retries == 5
        assert manager.config.max_cost_per_search == 200.0
        assert len(manager.retry_history) == 0
        assert manager.total_retry_cost == 0.0
    
    def test_simplified_name_strategy(self):
        """Test the simplified name retry strategy."""
        manager = LinkedInRetryManager()
        
        search = {"company_name": "Test Company SAS"}
        result = manager._retry_with_simplified_name(search, 1)
        
        assert result is not None
        assert "SAS" not in result["company_name"]
        assert result["company_name"] == "Test Company"
    
    def test_alternative_positions_strategy(self):
        """Test the alternative positions retry strategy."""
        manager = LinkedInRetryManager()
        
        search = {"company_name": "Test Company"}
        result = manager._retry_with_alternative_positions(search, 1)
        
        assert result is not None
        assert "filters" in result
        assert "positions" in result["filters"]
        assert len(result["filters"]["positions"]) > 0
    
    def test_broader_search_strategy(self):
        """Test the broader search retry strategy."""
        manager = LinkedInRetryManager()
        
        search = {"maxProfiles": 5, "filters": {"positions": ["CEO"]}}
        result = manager._retry_with_broader_search(search, 1)
        
        assert result is not None
        assert result["maxProfiles"] > 5
    
    def test_cost_estimation(self):
        """Test cost estimation for retries."""
        manager = LinkedInRetryManager()
        
        cost1 = manager._estimate_retry_cost(1)
        cost2 = manager._estimate_retry_cost(2)
        cost3 = manager._estimate_retry_cost(3)
        
        assert cost1 < cost2 < cost3
        assert all(cost <= 100.0 for cost in [cost1, cost2, cost3])  # Cost cap
    
    def test_google_maps_retry_manager(self):
        """Test Google Maps retry manager."""
        manager = GoogleMapsRetryManager()
        
        search = {"address": "123 Main St apt 4B"}  # lowercase apt should trigger pattern
        result = manager._simplify_address(search)
        
        assert result is not None
        assert "apt 4B" not in result["address"]


class TestCostManager:
    """Test the cost management functionality."""
    
    def test_cost_manager_initialization(self):
        """Test cost manager initialization."""
        budget_config = BudgetConfig(total_daily_budget=500.0)
        manager = CostManager(budget_config)
        
        assert manager.budget_config.total_daily_budget == 500.0
        assert len(manager.cost_entries) == 0
        assert manager.current_costs[ScraperType.GOOGLE_PLACES] == 0.0
    
    def test_cost_estimation(self):
        """Test operation cost estimation."""
        manager = CostManager()
        
        # Test LinkedIn cost estimation
        linkedin_details = {"maxProfiles": 5, "searchTerms": ["Company A"]}
        linkedin_cost = manager.estimate_operation_cost(ScraperType.LINKEDIN_PREMIUM, linkedin_details)
        
        # Test Google Places cost estimation
        places_details = {"max_places_per_search": 10}
        places_cost = manager.estimate_operation_cost(ScraperType.GOOGLE_PLACES, places_details)
        
        assert linkedin_cost > places_cost  # LinkedIn should be more expensive
        assert linkedin_cost > 0
        assert places_cost > 0
    
    def test_operation_tracking(self):
        """Test operation start and completion tracking."""
        manager = CostManager()
        
        operation_details = {"test": True}
        entry = manager.track_operation_start(
            ScraperType.GOOGLE_PLACES, "test_op_123", operation_details
        )
        
        assert entry.operation_id == "test_op_123"
        assert entry.scraper_type == ScraperType.GOOGLE_PLACES
        assert len(manager.cost_entries) == 1
        
        # Complete the operation
        manager.track_operation_complete("test_op_123", actual_cost=15.0, success=True, results_count=5)
        
        assert entry.actual_cost == 15.0
        assert entry.success is True
    
    def test_budget_checks(self):
        """Test budget constraint checking."""
        budget_config = BudgetConfig(total_daily_budget=100.0)
        manager = CostManager(budget_config)
        
        # Simulate high existing costs
        manager.current_costs[ScraperType.GOOGLE_PLACES] = 95.0
        
        # Test operation that would definitely exceed budget
        operation_details = {"addresses_count": 100, "max_places_per_search": 25}
        cost_check = manager.pre_operation_check(ScraperType.GOOGLE_PLACES, operation_details)
        
        # Should be blocked due to budget constraints or at least have warnings
        assert cost_check["can_proceed"] is False or len(cost_check["warnings"]) > 0
    
    def test_real_time_costs(self):
        """Test real-time cost reporting."""
        manager = CostManager()
        
        # Add some costs
        manager.current_costs[ScraperType.GOOGLE_PLACES] = 50.0
        manager.current_costs[ScraperType.LINKEDIN_PREMIUM] = 100.0
        manager.operation_counts[ScraperType.GOOGLE_PLACES] = 10
        
        costs = manager.get_real_time_costs()
        
        assert costs["total_current_cost"] == 150.0
        assert costs["operation_counts"]["google_places"] == 10
        assert "cost_efficiency_score" in costs  # Correct key name
    
    def test_create_cost_manager_from_config(self):
        """Test creating cost manager from job configuration."""
        config = {
            "apify": {
                "budget": {
                    "total_daily_budget": 750.0,
                    "warning_threshold": 0.75
                }
            }
        }
        
        manager = create_cost_manager(config)
        assert manager.budget_config.total_daily_budget == 750.0
        assert manager.budget_config.warning_threshold == 0.75


class TestDynamicConfiguration:
    """Test the dynamic configuration functionality."""
    
    def test_dynamic_config_manager_initialization(self):
        """Test dynamic configuration manager initialization."""
        manager = DynamicConfigurationManager()
        
        assert len(manager.profiles) == 5  # All budget modes
        assert manager.current_profile is None
        assert len(manager.configuration_history) == 0
    
    def test_budget_mode_selection(self):
        """Test budget mode selection logic."""
        manager = DynamicConfigurationManager()
        
        # Test minimal budget
        mode = manager._select_budget_mode(30.0, {})
        assert mode == BudgetMode.MINIMAL
        
        # Test balanced budget
        mode = manager._select_budget_mode(400.0, {})
        assert mode == BudgetMode.BALANCED
        
        # Test unlimited budget
        mode = manager._select_budget_mode(2000.0, {})
        assert mode == BudgetMode.UNLIMITED
    
    def test_priority_customization(self):
        """Test configuration customization based on priorities."""
        manager = DynamicConfigurationManager()
        
        base_profile = manager.profiles[BudgetMode.BALANCED]
        priorities = [ConfigPriority.CONTACTS, ConfigPriority.EXECUTIVES]
        
        customized = manager._customize_for_priorities(base_profile, priorities, 800.0)
        
        # Should boost contact and LinkedIn settings
        assert customized.google_maps_contacts_config["enabled"] is True
        assert customized.linkedin_premium_config["enabled"] is True
    
    def test_optimal_configuration_determination(self):
        """Test optimal configuration determination."""
        manager = DynamicConfigurationManager()
        
        base_config = {"apify": {"enabled": True}}
        priorities = [ConfigPriority.QUALITY]
        context = {"dry_run": False, "total_addresses": 25}
        
        config = manager.determine_optimal_configuration(
            base_config, 500.0, priorities, context
        )
        
        assert "apify" in config
        assert config["apify"]["max_addresses"] > 0
        assert len(manager.configuration_history) == 1
    
    def test_runtime_adjustment(self):
        """Test runtime configuration adjustment."""
        manager = DynamicConfigurationManager()
        manager.current_profile = manager.profiles[BudgetMode.BALANCED]
        
        current_config = {
            "apify": {
                "max_addresses": 25,
                "linkedin_premium": {"enabled": True, "max_linkedin_searches": 20}
            }
        }
        
        # Simulate poor performance
        performance_metrics = {"avg_operation_time_seconds": 400, "success_rate": 0.6}
        cost_metrics = {"cost_per_result": 12.0, "remaining_budget": 50.0}
        
        adjusted = manager.adjust_configuration_runtime(
            current_config, performance_metrics, cost_metrics
        )
        
        # Should reduce scope due to high cost and poor performance
        assert adjusted["apify"]["max_addresses"] < 25
    
    def test_create_dynamic_config_manager(self):
        """Test creating dynamic config manager."""
        manager = create_dynamic_config_manager()
        assert isinstance(manager, DynamicConfigurationManager)


class TestIndustryOptimizer:
    """Test the industry optimization functionality."""
    
    def test_industry_optimizer_initialization(self):
        """Test industry optimizer initialization."""
        optimizer = IndustryOptimizer()
        
        assert len(optimizer.industry_profiles) >= len(IndustryCategory)
        assert len(optimizer.naf_to_industry) > 0
        assert len(optimizer.optimization_history) == 0
    
    def test_industry_detection_by_naf(self):
        """Test industry detection using NAF codes."""
        optimizer = IndustryOptimizer()
        
        # Test technology NAF code
        business_data = {"naf_code": "6201Z", "company_name": "Tech Solutions"}
        industry, confidence = optimizer.detect_industry(business_data)
        
        assert industry == IndustryCategory.TECHNOLOGY
        assert confidence > 0.5
    
    def test_industry_detection_by_name(self):
        """Test industry detection using company name."""
        optimizer = IndustryOptimizer()
        
        # Test healthcare company
        business_data = {"company_name": "City Medical Center", "description": "healthcare services"}
        industry, confidence = optimizer.detect_industry(business_data)
        
        assert industry == IndustryCategory.HEALTHCARE
        assert confidence > 0.0
    
    def test_configuration_optimization(self):
        """Test industry-specific configuration optimization."""
        optimizer = IndustryOptimizer()
        
        base_config = {
            "apify": {
                "google_places": {"enabled": True, "max_places_per_search": 10},
                "linkedin_premium": {"enabled": True, "max_linkedin_searches": 10}
            }
        }
        
        optimized = optimizer.optimize_configuration(
            base_config, 
            IndustryCategory.TECHNOLOGY, 
            OptimizationStrategy.EXECUTIVE_FOCUSED,
            {}
        )
        
        # Technology + executive focus should boost LinkedIn
        linkedin_config = optimized["apify"]["linkedin_premium"]
        assert linkedin_config["enabled"] is True
        assert linkedin_config["max_linkedin_searches"] >= 10
    
    def test_optimization_strategy_suggestion(self):
        """Test optimization strategy suggestion."""
        optimizer = IndustryOptimizer()
        
        business_context = {"requires_executives": True, "budget_constrained": False}
        strategy, reason = optimizer.suggest_optimization_strategy(
            IndustryCategory.FINANCE, business_context, 800.0
        )
        
        assert strategy == OptimizationStrategy.EXECUTIVE_FOCUSED
        assert "executive" in reason.lower()
    
    def test_industry_insights(self):
        """Test industry insights generation."""
        optimizer = IndustryOptimizer()
        
        insights = optimizer.get_industry_insights(IndustryCategory.TECHNOLOGY)
        
        assert insights["industry"] == "technology"
        assert "scraper_priorities" in insights
        assert "recommended_linkedin_positions" in insights
        assert len(insights["key_search_keywords"]) > 0
    
    def test_create_industry_optimizer(self):
        """Test creating industry optimizer."""
        optimizer = create_industry_optimizer()
        assert isinstance(optimizer, IndustryOptimizer)


class TestIntegration:
    """Test integration between components."""
    
    def test_cost_manager_and_dynamic_config_integration(self):
        """Test integration between cost manager and dynamic configuration."""
        cost_manager = CostManager()
        dynamic_config = DynamicConfigurationManager(cost_manager)
        
        assert dynamic_config.cost_manager is cost_manager
    
    def test_end_to_end_configuration_flow(self):
        """Test complete configuration flow."""
        # Initialize all components
        cost_manager = CostManager()
        dynamic_config = DynamicConfigurationManager(cost_manager)
        industry_optimizer = IndustryOptimizer()
        
        # Detect industry
        business_data = {"naf_code": "6201Z", "company_name": "TechCorp"}
        industry, confidence = industry_optimizer.detect_industry(business_data)
        
        # Determine optimal configuration
        base_config = {"apify": {"enabled": True}}
        priorities = [ConfigPriority.EXECUTIVES]
        context = {"total_addresses": 20}
        
        config = dynamic_config.determine_optimal_configuration(
            base_config, 600.0, priorities, context
        )
        
        # Apply industry optimization
        optimized_config = industry_optimizer.optimize_configuration(
            config, industry, OptimizationStrategy.EXECUTIVE_FOCUSED, context
        )
        
        # Verify the complete flow worked
        assert "apify" in optimized_config
        assert optimized_config["apify"]["linkedin_premium"]["enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__])