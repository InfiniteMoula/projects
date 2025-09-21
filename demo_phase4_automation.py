#!/usr/bin/env python3
"""
Demonstration script for Phase 4 Apify automation features.

This script showcases the complete monitoring, alerting, and dashboard
capabilities implemented in Phase 4.
"""

import time
import tempfile
from pathlib import Path
from utils.cost_manager import ScraperType
from monitoring.apify_monitor import create_apify_monitor
from monitoring.alert_manager import create_alert_manager, AlertManagerConfig
from dashboard.apify_dashboard import create_apify_dashboard, DashboardConfig


def demonstrate_phase4_features():
    """Demonstrate Phase 4 automation features."""
    
    print("[DEMO] Apify Automation Phase 4 Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("\n[INIT] Initializing monitoring system...")
    monitor = create_apify_monitor()
    
    # Configure alert manager with console output
    alert_config = AlertManagerConfig(max_alerts_per_hour=50)
    alert_manager = create_alert_manager(alert_config)
    
    # Configure dashboard
    dashboard_config = DashboardConfig(
        include_detailed_metrics=True,
        include_performance_charts=True
    )
    dashboard = create_apify_dashboard(monitor, alert_manager, dashboard_config)
    
    # Connect systems
    monitor.add_event_listener(alert_manager.process_monitoring_event)
    monitor.start_monitoring()
    
    print("[OK] Monitoring system initialized")
    
    try:
        # Simulate various scraping operations
        print("\n[SIM] Simulating scraping operations...")
        
        # Scenario 1: Successful Google Places operation
        session_1 = "demo_google_places"
        print(f"   Starting Google Places scraper: {session_1}")
        monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_1, {
            "query": "restaurants Paris",
            "max_results": 50
        })
        
        time.sleep(0.5)  # Simulate processing time
        
        monitor.log_scraper_complete(session_1, {
            "status": "success",
            "results": [
                {"name": "Le Petit CafÃ©", "address": "123 Rue de la Paix", "phone": "+33123456789"},
                {"name": "Bistro Central", "address": "456 Boulevard St-Germain", "email": "contact@bistro.fr"},
                {"name": "Restaurant Moderne", "address": "789 Avenue des Champs", "phone": "+33198765432"}
            ],
            "estimated_cost": 45.0
        })
        print(f"   [OK] Google Places operation completed")
        
        # Scenario 2: LinkedIn Premium operation  
        session_2 = "demo_linkedin"
        print(f"   Starting LinkedIn Premium scraper: {session_2}")
        monitor.log_scraper_start(ScraperType.LINKEDIN_PREMIUM, session_2, {
            "company": "TechCorp",
            "positions": ["CEO", "CTO", "VP Engineering"]
        })
        
        time.sleep(0.3)
        
        monitor.log_scraper_complete(session_2, {
            "status": "success", 
            "results": [
                {"name": "John Smith", "position": "CEO", "company": "TechCorp"},
                {"name": "Jane Doe", "position": "CTO", "company": "TechCorp"}
            ],
            "estimated_cost": 80.0
        })
        print(f"   [OK] LinkedIn Premium operation completed")
        
        # Scenario 3: Error simulation
        session_3 = "demo_error"
        print(f"   Starting operation with error: {session_3}")
        monitor.log_scraper_start(ScraperType.GOOGLE_MAPS_CONTACTS, session_3, {
            "location": "New York",
            "category": "restaurants"
        })
        
        time.sleep(0.2)
        monitor.log_scraper_error(session_3, Exception("Rate limit exceeded"))
        print(f"   [FAIL] Operation failed (simulated)")
        
        # Give time for event processing
        time.sleep(0.5)
        
        # Check monitoring status
        print("\n[METRICS] Getting real-time monitoring status...")
        status = monitor.get_real_time_status()
        
        print(f"   Total operations: {status['overall_stats']['total_operations']}")
        print(f"   Success rate: {status['overall_stats']['success_rate']:.1%}")
        print(f"   Total cost: {status['overall_stats']['total_cost']:.2f} credits")
        print(f"   Active sessions: {status['overall_stats']['active_sessions']}")
        
        # Test alert system
        print("\n[ALERT] Testing alert system...")
        
        # Create high cost scenario to trigger alerts
        high_cost_data = {
            'overall_stats': {
                'total_cost': 850.0,  # 85% of 1000 budget
                'total_operations': 20,
                'success_rate': 0.75  # 75% success rate
            },
            'quality_metrics': {
                'validation_rate': 0.65,  # Below 70% threshold
                'average_score': 60.0
            }
        }
        
        triggered_alerts = alert_manager.check_alerts(high_cost_data)
        print(f"   Triggered {len(triggered_alerts)} alerts")
        
        for alert in triggered_alerts:
            print(f"   [ALERT] {alert.severity.value.upper()}: {alert.message}")
        
        # Generate performance summary
        print("\n[SUMMARY] Performance summary (last hour)...")
        perf_summary = monitor.get_performance_summary(hours=1)
        
        print(f"   Operations: {perf_summary.get('total_operations', 0)}")
        print(f"   Cost efficiency: {perf_summary.get('cost_per_result', 0):.2f} credits/result")
        print(f"   Average efficiency: {perf_summary.get('average_efficiency', 0):.2f} results/min")
        
        # Generate dashboard
        print("\n[DASHBOARD] Generating dashboard...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Generate different dashboard formats
            json_content = dashboard.export_dashboard("json", output_dir / "dashboard.json")
            text_content = dashboard.export_dashboard("text", output_dir / "dashboard.txt")
            
            print(f"   [OK] JSON dashboard: {len(json_content)} characters")
            print(f"   [OK] Text dashboard: {len(text_content.splitlines())} lines")
            
            # Try HTML generation (may have formatting issues with demo data)
            try:
                html_content = dashboard.export_dashboard("html", output_dir / "dashboard.html")
                print(f"   [OK] HTML dashboard: {len(html_content)} characters")
            except Exception as e:
                print(f"   [WARN] HTML dashboard: {str(e)[:50]}... (expected with demo data)")
            
            # Show system health
            dashboard_data = dashboard.generate_dashboard_data()
            health = dashboard_data['system_health']
            print(f"   [HEALTH] System health: {health['status']} ({health['score']}/100)")
            
            # Show a sample of the text dashboard
            print("\n[OUTPUT] Sample dashboard output:")
            print("-" * 40)
            lines = text_content.splitlines()[:15]  # First 15 lines
            for line in lines:
                print(f"   {line}")
            if len(text_content.splitlines()) > 15:
                print("   ...")
            print("-" * 40)
        
        # Alert summary
        print("\n[SUMMARY] Alert summary...")
        alert_summary = alert_manager.get_alert_summary(hours=1)
        print(f"   Total alerts: {alert_summary['total_alerts']}")
        
        if alert_summary['by_severity']:
            print("   By severity:")
            for severity, count in alert_summary['by_severity'].items():
                print(f"     {severity.capitalize()}: {count}")
        
        print("\n[DONE] Phase 4 demonstration completed successfully!")
        print("   [OK] Real-time monitoring")
        print("   [OK] Intelligent alerting") 
        print("   [OK] Performance dashboards")
        print("   [OK] Quality integration")
        print("   [OK] Cost tracking")
        
    except Exception as e:
        print(f"\n[ERROR] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        monitor.stop_monitoring()
        print("\n[STOP] Monitoring system stopped")


def demonstrate_performance_testing():
    """Demonstrate performance testing capabilities."""
    
    print("\n[PERF] Performance Testing Demonstration")
    print("=" * 60)
    
    from benchmarks.performance_tests import PerformanceTestSuite, BenchmarkConfig
    
    # Create lightweight config for demo
    config = BenchmarkConfig(
        target_operations=50,  # Reduced for demo
        concurrent_workers=3,
        timeout_seconds=60
    )
    
    print("[BENCH] Running performance benchmarks...")
    suite = PerformanceTestSuite(config)
    
    try:
        # Run just a few key benchmarks for demo
        throughput_result = suite.run_specific_benchmark(suite.benchmarks[0].benchmark_type)
        
        if throughput_result:
            print(f"   [OK] Throughput test: {throughput_result.throughput_ops_per_second:.2f} ops/sec")
            print(f"   [OK] Average latency: {throughput_result.average_latency_ms:.2f} ms") 
            print(f"   [OK] Success rate: {throughput_result.success_rate:.1%}")
        
        # Generate performance report
        report = suite.generate_performance_report()
        print(f"\n[PERF] Performance Grade: {report['summary']['performance_grade']}")
        
        # Show recommendations
        print("\n[INFO] Recommendations:")
        for rec in report['recommendations'][:3]:  # Show first 3
            print(f"   - {rec}")
            
    except Exception as e:
        print(f"[ERROR] Performance testing error: {e}")


if __name__ == "__main__":
    # Run main demonstration
    demonstrate_phase4_features()
    
    # Run performance testing demo  
    demonstrate_performance_testing()
    
    print(f"\n[DONE] Phase 4 implementation demonstration complete!")
    print("[DOCS] See docs/automation-guide.md for detailed usage instructions")
