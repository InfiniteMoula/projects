#!/usr/bin/env python3
"""
Comprehensive end-to-end tests for Apify automation features.

This module provides integration tests that validate the complete automation
pipeline from monitoring to alerting to dashboard generation.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from pathlib import Path
import tempfile
import json

from monitoring.apify_monitor import ApifyMonitor, MonitoringEvent, MonitoringEventType, create_apify_monitor
from monitoring.alert_manager import AlertManager, AlertRule, AlertSeverity, AlertChannel, create_alert_manager
from dashboard.apify_dashboard import ApifyDashboard, DashboardConfig, create_apify_dashboard
from utils.cost_manager import CostManager, ScraperType, BudgetConfig
from utils.quality_controller import GoogleMapsQualityController, LinkedInQualityController


class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    def test_monitor_initialization(self):
        """Test ApifyMonitor initialization with dependencies."""
        cost_manager = CostManager()
        monitor = ApifyMonitor(cost_manager)
        
        assert monitor.cost_manager is cost_manager
        assert 'google_maps' in monitor.quality_controllers
        assert 'linkedin' in monitor.quality_controllers
        assert monitor.monitoring_active is False
        assert len(monitor.event_listeners) == 0
    
    def test_scraper_lifecycle_monitoring(self):
        """Test complete scraper operation lifecycle monitoring."""
        monitor = create_apify_monitor()
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        
        # Log scraper start
        session_id = "test_session_001"
        scraper_type = ScraperType.GOOGLE_PLACES
        operation_details = {"query": "restaurants Paris", "max_results": 100}
        
        monitor.log_scraper_start(scraper_type, session_id, operation_details)
        
        # Verify session data
        assert session_id in monitor.session_data
        assert monitor.session_data[session_id]['scraper_type'] == scraper_type
        
        # Simulate scraper completion
        results = {
            "status": "success",
            "results": [
                {"name": "Restaurant A", "address": "123 Rue A", "phone": "+33123456789"},
                {"name": "Restaurant B", "address": "456 Rue B", "email": "contact@restaurantb.fr"}
            ],
            "estimated_cost": 50.0
        }
        
        monitor.log_scraper_complete(session_id, results)
        
        # Verify session cleanup
        assert session_id not in monitor.session_data
        
        # Check metrics update
        stats = monitor.metrics.scraper_stats[scraper_type]
        assert stats['total_operations'] == 1
        assert stats['successful_operations'] == 1
        assert stats['total_cost'] == 50.0
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
    
    def test_error_handling_in_monitoring(self):
        """Test error handling in monitoring system."""
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        session_id = "error_session"
        scraper_type = ScraperType.LINKEDIN_PREMIUM
        
        # Log scraper start
        monitor.log_scraper_start(scraper_type, session_id, {"query": "tech companies"})
        
        # Log error
        error = Exception("API rate limit exceeded")
        monitor.log_scraper_error(session_id, error)
        
        # Verify error tracking
        stats = monitor.metrics.scraper_stats[scraper_type]
        assert stats['failed_operations'] == 1
        
        # Verify session cleanup
        assert session_id not in monitor.session_data
        
        monitor.stop_monitoring()
    
    def test_event_listeners(self):
        """Test event listener functionality."""
        monitor = create_apify_monitor()
        
        events_received = []
        
        def test_listener(event):
            events_received.append(event)
        
        monitor.add_event_listener(test_listener)
        
        # Start monitoring and generate events
        monitor.start_monitoring()
        
        session_id = "listener_test"
        monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
        
        # Wait for event processing
        time.sleep(0.1)
        
        # Verify event was received
        assert len(events_received) == 1
        assert events_received[0].event_type == MonitoringEventType.SCRAPER_START
        
        monitor.stop_monitoring()
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        # Simulate multiple operations
        for i in range(3):
            session_id = f"perf_test_{i}"
            monitor.log_scraper_start(ScraperType.GOOGLE_MAPS_CONTACTS, session_id, {})
            
            # Simulate different performance results
            results = {
                "status": "success",
                "results": [f"result_{j}" for j in range(5 + i)],  # Varying result counts
                "estimated_cost": 25.0 + (i * 10)  # Varying costs
            }
            
            time.sleep(0.01)  # Small delay to simulate processing time
            monitor.log_scraper_complete(session_id, results)
        
        # Get performance summary
        summary = monitor.get_performance_summary(hours=1)
        
        assert summary['total_operations'] == 3
        assert summary['total_cost'] == 75.0 + 30.0  # 105.0
        assert summary['total_results'] == 5 + 6 + 7  # 18
        
        monitor.stop_monitoring()


class TestAlertingSystem:
    """Test alerting system functionality."""
    
    def test_alert_manager_initialization(self):
        """Test AlertManager initialization with default rules."""
        alert_manager = create_alert_manager()
        
        # Verify default rules are loaded
        assert len(alert_manager.rules) > 0
        assert "high_cost_usage" in alert_manager.rules
        assert "critical_cost_usage" in alert_manager.rules
        assert "low_quality_rate" in alert_manager.rules
    
    def test_custom_alert_rules(self):
        """Test custom alert rule creation and management."""
        alert_manager = create_alert_manager()
        
        # Add custom rule
        custom_rule = AlertRule(
            name="test_rule",
            description="Test alert rule",
            condition=lambda data: data.get('test_value', 0) > 100,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG]
        )
        
        alert_manager.add_rule(custom_rule)
        assert "test_rule" in alert_manager.rules
        
        # Test rule enablement/disablement
        alert_manager.disable_rule("test_rule")
        assert alert_manager.rules["test_rule"].enabled is False
        
        alert_manager.enable_rule("test_rule")
        assert alert_manager.rules["test_rule"].enabled is True
        
        # Remove rule
        alert_manager.remove_rule("test_rule")
        assert "test_rule" not in alert_manager.rules
    
    def test_alert_triggering(self):
        """Test alert triggering based on monitoring data."""
        alert_manager = create_alert_manager()
        
        # Test data that should trigger high cost usage alert
        monitoring_data = {
            'overall_stats': {
                'total_cost': 850.0,  # 85% of 1000 budget
                'total_operations': 100,
                'success_rate': 0.9
            },
            'quality_metrics': {
                'validation_rate': 0.8,
                'average_score': 75.0
            }
        }
        
        # Check alerts
        triggered_alerts = alert_manager.check_alerts(monitoring_data)
        
        # Should trigger high_cost_usage alert (>80%)
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].rule_name == "high_cost_usage"
        assert triggered_alerts[0].severity == AlertSeverity.HIGH
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        alert_manager = create_alert_manager()
        
        # Set short cooldown for testing
        alert_manager.rules["high_cost_usage"].cooldown_seconds = 1
        
        monitoring_data = {
            'overall_stats': {
                'total_cost': 850.0,
                'total_operations': 100,
                'success_rate': 0.9
            }
        }
        
        # First check should trigger alert
        alerts1 = alert_manager.check_alerts(monitoring_data)
        assert len(alerts1) == 1
        
        # Immediate second check should not trigger (cooldown)
        alerts2 = alert_manager.check_alerts(monitoring_data)
        assert len(alerts2) == 0
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Third check should trigger again
        alerts3 = alert_manager.check_alerts(monitoring_data)
        assert len(alerts3) == 1
    
    def test_monitoring_event_processing(self):
        """Test processing of monitoring events."""
        alert_manager = create_alert_manager()
        
        # Create cost alert event
        cost_event = MonitoringEvent(
            timestamp=time.time(),
            event_type=MonitoringEventType.COST_ALERT,
            scraper_type=ScraperType.GOOGLE_PLACES,
            data={
                'alert_type': 'CRITICAL_BUDGET',
                'message': 'Critical budget usage: 95%',
                'current_cost': 950.0,
                'budget_limit': 1000.0
            }
        )
        
        alert_manager.process_monitoring_event(cost_event)
        
        # Verify alert was created
        assert len(alert_manager.alerts) > 0
        latest_alert = alert_manager.alerts[-1]
        assert latest_alert.rule_name == "cost_threshold"
        assert latest_alert.severity == AlertSeverity.HIGH  # CRITICAL in message should map to HIGH
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        alert_manager = create_alert_manager()
        
        # Add some test alerts manually
        current_time = time.time()
        test_alerts = [
            Mock(
                timestamp=current_time - 1800,  # 30 minutes ago
                rule_name="test_rule_1",
                severity=AlertSeverity.HIGH,
                acknowledged=False,
                resolved=False
            ),
            Mock(
                timestamp=current_time - 3600,  # 1 hour ago
                rule_name="test_rule_2",
                severity=AlertSeverity.MEDIUM,
                acknowledged=True,
                resolved=False
            )
        ]
        
        alert_manager.alert_history.extend(test_alerts)
        
        # Get summary
        summary = alert_manager.get_alert_summary(hours=2)
        
        assert summary['total_alerts'] == 2
        assert summary['by_severity']['high'] == 1
        assert summary['by_severity']['medium'] == 1


class TestDashboardGeneration:
    """Test dashboard generation and visualization."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization with dependencies."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        assert dashboard.monitor is monitor
        assert dashboard.alert_manager is alert_manager
        assert dashboard.config is not None
    
    def test_dashboard_data_generation(self):
        """Test comprehensive dashboard data generation."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        # Generate some test data
        monitor.start_monitoring()
        
        # Simulate operations
        session_id = "dashboard_test"
        monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
        monitor.log_scraper_complete(session_id, {
            "status": "success",
            "results": ["result1", "result2"],
            "estimated_cost": 30.0
        })
        
        # Generate dashboard data
        dashboard_data = dashboard.generate_dashboard_data()
        
        # Verify structure
        assert 'timestamp' in dashboard_data
        assert 'key_metrics' in dashboard_data
        assert 'monitoring_status' in dashboard_data
        assert 'performance' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'system_health' in dashboard_data
        
        # Verify key metrics
        metrics = dashboard_data['key_metrics']
        assert 'success_rate_percent' in metrics
        assert 'cost_per_result' in metrics
        assert 'results_per_minute' in metrics
        
        monitor.stop_monitoring()
    
    def test_text_dashboard_generation(self):
        """Test text dashboard format generation."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        text_dashboard = dashboard.generate_text_dashboard()
        
        # Verify text dashboard content
        assert "APIFY AUTOMATION DASHBOARD" in text_dashboard
        assert "SYSTEM HEALTH:" in text_dashboard
        assert "KEY METRICS:" in text_dashboard
        assert "24-HOUR PERFORMANCE:" in text_dashboard
    
    def test_html_dashboard_generation(self):
        """Test HTML dashboard format generation."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        html_dashboard = dashboard.generate_html_dashboard()
        
        # Verify HTML structure
        assert "<!DOCTYPE html>" in html_dashboard
        assert "<title>Apify Automation Dashboard</title>" in html_dashboard
        assert "System Health Score" in html_dashboard
        assert "Success Rate" in html_dashboard
    
    def test_dashboard_export(self):
        """Test dashboard export functionality."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test JSON export
            json_file = temp_path / "dashboard.json"
            json_content = dashboard.export_dashboard("json", json_file)
            
            assert json_file.exists()
            assert json.loads(json_content)  # Valid JSON
            
            # Test HTML export
            html_file = temp_path / "dashboard.html"
            html_content = dashboard.export_dashboard("html", html_file)
            
            assert html_file.exists()
            assert "<!DOCTYPE html>" in html_content
            
            # Test text export
            text_file = temp_path / "dashboard.txt"
            text_content = dashboard.export_dashboard("text", text_file)
            
            assert text_file.exists()
            assert "APIFY AUTOMATION DASHBOARD" in text_content


class TestEndToEndIntegration:
    """Test complete end-to-end automation workflows."""
    
    def test_complete_monitoring_pipeline(self):
        """Test complete monitoring pipeline integration."""
        # Initialize all components
        cost_manager = CostManager()
        monitor = ApifyMonitor(cost_manager)
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        # Connect alert manager to monitor events
        monitor.add_event_listener(alert_manager.process_monitoring_event)
        
        monitor.start_monitoring()
        
        # Simulate a complete scraping workflow
        session_id = "e2e_test_session"
        scraper_type = ScraperType.GOOGLE_MAPS_CONTACTS
        
        # Step 1: Start scraper
        monitor.log_scraper_start(scraper_type, session_id, {
            "query": "restaurants New York",
            "max_results": 50
        })
        
        # Step 2: Complete scraper with high cost (should trigger alert)
        results = {
            "status": "success",
            "results": [{"name": f"Restaurant {i}"} for i in range(20)],
            "estimated_cost": 500.0  # High cost
        }
        
        monitor.log_scraper_complete(session_id, results)
        
        # Step 3: Check for alerts
        monitoring_status = monitor.get_real_time_status()
        triggered_alerts = alert_manager.check_alerts(monitoring_status)
        
        # Step 4: Generate dashboard
        dashboard_data = dashboard.generate_dashboard_data()
        
        # Verify end-to-end flow
        assert monitoring_status['overall_stats']['total_operations'] == 1
        assert monitoring_status['overall_stats']['total_cost'] == 500.0
        assert dashboard_data['key_metrics']['total_operations_24h'] == 1
        
        # Verify system health assessment
        health = dashboard_data['system_health']
        assert 'score' in health
        assert 'status' in health
        
        monitor.stop_monitoring()
    
    def test_multi_scraper_coordination(self):
        """Test coordination between multiple scraper types."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        
        monitor.start_monitoring()
        
        # Simulate multiple concurrent scrapers
        sessions = [
            ("google_session", ScraperType.GOOGLE_PLACES),
            ("linkedin_session", ScraperType.LINKEDIN_PREMIUM),
            ("maps_session", ScraperType.GOOGLE_MAPS_CONTACTS)
        ]
        
        # Start all scrapers
        for session_id, scraper_type in sessions:
            monitor.log_scraper_start(scraper_type, session_id, {})
        
        # Complete scrapers with different outcomes
        outcomes = [
            ("google_session", {"status": "success", "results": [1, 2, 3], "estimated_cost": 100}),
            ("linkedin_session", {"status": "success", "results": [1, 2], "estimated_cost": 200}),
            ("maps_session", {"status": "success", "results": [1, 2, 3, 4], "estimated_cost": 150})
        ]
        
        for session_id, result in outcomes:
            monitor.log_scraper_complete(session_id, result)
        
        # Verify metrics for all scrapers
        status = monitor.get_real_time_status()
        scraper_stats = status['scraper_stats']
        
        assert len(scraper_stats) == 3
        assert scraper_stats[ScraperType.GOOGLE_PLACES]['total_cost'] == 100
        assert scraper_stats[ScraperType.LINKEDIN_PREMIUM]['total_cost'] == 200
        assert scraper_stats[ScraperType.GOOGLE_MAPS_CONTACTS]['total_cost'] == 150
        
        monitor.stop_monitoring()
    
    def test_error_recovery_and_monitoring(self):
        """Test error recovery and monitoring capabilities."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        
        monitor.add_event_listener(alert_manager.process_monitoring_event)
        monitor.start_monitoring()
        
        # Simulate failed operations
        failed_sessions = ["fail_1", "fail_2", "fail_3"]
        
        for session_id in failed_sessions:
            monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
            monitor.log_scraper_error(session_id, Exception("Network timeout"))
        
        # Simulate successful operation
        success_session = "success_1"
        monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, success_session, {})
        monitor.log_scraper_complete(success_session, {
            "status": "success",
            "results": [1, 2, 3],
            "estimated_cost": 50
        })
        
        # Check error tracking
        status = monitor.get_real_time_status()
        stats = status['scraper_stats'][ScraperType.GOOGLE_PLACES]
        
        assert stats['failed_operations'] == 3
        assert stats['successful_operations'] == 1
        assert status['overall_stats']['success_rate'] == 0.25  # 1/4
        
        # Verify alerts for high error rate
        monitoring_data = status
        alerts = alert_manager.check_alerts(monitoring_data)
        
        # Should trigger high error rate alert
        error_alerts = [a for a in alerts if "error" in a.rule_name]
        assert len(error_alerts) > 0
        
        monitor.stop_monitoring()


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_monitoring_performance_overhead(self):
        """Test that monitoring doesn't add significant overhead."""
        import time
        
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        # Measure time for operations with monitoring
        start_time = time.time()
        
        for i in range(100):
            session_id = f"perf_test_{i}"
            monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
            monitor.log_scraper_complete(session_id, {
                "status": "success",
                "results": [1, 2],
                "estimated_cost": 10
            })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 100 operations should complete in reasonable time (< 1 second)
        assert total_time < 1.0, f"Monitoring overhead too high: {total_time}s for 100 operations"
        
        monitor.stop_monitoring()
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization in long-running monitoring."""
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        # Generate many events
        for i in range(5000):
            session_id = f"memory_test_{i}"
            monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
            monitor.log_scraper_complete(session_id, {
                "status": "success",
                "results": [1],
                "estimated_cost": 5
            })
        
        # Check that event deque is limited
        assert len(monitor.metrics.events) <= 10000  # Max size enforced
        assert len(monitor.performance_history) <= 1000  # Max size enforced
        
        monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])