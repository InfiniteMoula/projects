# Apify Automation Complete Guide

This comprehensive guide covers the complete Apify automation system implemented in Phase 4, including monitoring, alerting, dashboard visualization, and performance optimization.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Monitoring System](#monitoring-system)
5. [Alerting Framework](#alerting-framework)
6. [Dashboard Visualization](#dashboard-visualization)
7. [Performance Testing](#performance-testing)
8. [Integration Examples](#integration-examples)
9. [Configuration Guide](#configuration-guide)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Overview

The Apify automation system provides comprehensive monitoring, alerting, and visualization capabilities for LinkedIn and Google Maps scraping operations. The system is designed for production environments with real-time monitoring, intelligent alerting, and performance optimization.

### Key Features

- **Real-time Monitoring**: Track scraper operations, costs, and quality metrics in real-time
- **Intelligent Alerting**: Customizable alert rules with multiple notification channels
- **Performance Dashboards**: JSON, HTML, and text-based dashboards with auto-refresh
- **Quality Integration**: Seamless integration with existing quality control framework
- **Cost Management**: Advanced cost tracking and budget enforcement
- **Performance Benchmarking**: Comprehensive performance testing and optimization

## Architecture

The automation system consists of four main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ApifyMonitor  │    │  AlertManager   │    │ ApifyDashboard  │
│                 │    │                 │    │                 │
│ • Event tracking│    │ • Rule engine   │    │ • Visualization │
│ • Quality checks│    │ • Notifications │    │ • Auto-reporting│
│ • Performance   │    │ • Correlations  │    │ • Multi-format  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │            Integration Layer                    │
         │                                                 │
         │  ┌─────────────┐  ┌─────────────┐               │
         │  │CostManager  │  │QualityCtrl  │               │
         │  └─────────────┘  └─────────────┘               │
         └─────────────────────────────────────────────────┘
```

### Component Overview

- **ApifyMonitor**: Central monitoring hub that tracks all scraper operations
- **AlertManager**: Advanced alerting system with rule-based notifications
- **ApifyDashboard**: Comprehensive dashboard generation and visualization
- **Integration Layer**: Seamless integration with existing cost and quality systems

## Quick Start

### Basic Setup

```python
from monitoring.apify_monitor import create_apify_monitor
from monitoring.alert_manager import create_alert_manager
from dashboard.apify_dashboard import create_apify_dashboard
from utils.cost_manager import CostManager

# Initialize components
cost_manager = CostManager()
monitor = create_apify_monitor(cost_manager)
alert_manager = create_alert_manager()
dashboard = create_apify_dashboard(monitor, alert_manager)

# Start monitoring
monitor.start_monitoring()

# Connect alert manager to monitoring events
monitor.add_event_listener(alert_manager.process_monitoring_event)
```

### Basic Monitoring Usage

```python
from utils.cost_manager import ScraperType

# Start a scraper operation
session_id = "my_scraper_session"
monitor.log_scraper_start(
    ScraperType.GOOGLE_PLACES, 
    session_id, 
    {"query": "restaurants Paris", "max_results": 100}
)

# Complete the operation
results = {
    "status": "success",
    "results": [{"name": "Restaurant A", "address": "123 Rue A"}],
    "estimated_cost": 50.0
}
monitor.log_scraper_complete(session_id, results)

# Get real-time status
status = monitor.get_real_time_status()
print(f"Total operations: {status['overall_stats']['total_operations']}")
print(f"Success rate: {status['overall_stats']['success_rate']:.1%}")
```

## Monitoring System

### ApifyMonitor Features

The `ApifyMonitor` class provides comprehensive monitoring capabilities:

#### Real-time Event Tracking

```python
# Supported event types
from monitoring.apify_monitor import MonitoringEventType

# Events are automatically generated for:
# - SCRAPER_START: When a scraper operation begins
# - SCRAPER_COMPLETE: When a scraper operation completes
# - SCRAPER_ERROR: When a scraper operation fails
# - QUALITY_CHECK: When quality validation occurs
# - COST_ALERT: When cost thresholds are exceeded
# - PERFORMANCE_METRIC: Performance milestone events
```

#### Quality Integration

The monitor automatically integrates with the quality control framework:

```python
# Quality checks are performed automatically on completion
# Results include:
# - Validation rate
# - Quality scores
# - Field coverage analysis
# - Common issues detection

# Access quality metrics
status = monitor.get_real_time_status()
quality_metrics = status['quality_metrics']
print(f"Validation rate: {quality_metrics['validation_rate']:.1%}")
print(f"Average score: {quality_metrics['average_score']}")
```

#### Performance Tracking

```python
# Get performance summary
perf_summary = monitor.get_performance_summary(hours=24)
print(f"Total operations: {perf_summary['total_operations']}")
print(f"Cost efficiency: {perf_summary['cost_per_result']:.2f} credits/result")
print(f"Average efficiency: {perf_summary['average_efficiency']:.2f} results/min")

# Scraper-specific breakdown
for scraper_type, stats in perf_summary['scraper_breakdown'].items():
    print(f"{scraper_type}: {stats['avg_efficiency']:.2f} results/min")
```

### Event Listeners

You can add custom event listeners to react to monitoring events:

```python
def custom_event_handler(event):
    if event.event_type == MonitoringEventType.SCRAPER_ERROR:
        print(f"Error detected: {event.data['error_message']}")
    elif event.event_type == MonitoringEventType.COST_ALERT:
        print(f"Cost alert: {event.data['message']}")

monitor.add_event_listener(custom_event_handler)
```

## Alerting Framework

### AlertManager Features

The `AlertManager` provides intelligent alerting with customizable rules:

#### Default Alert Rules

The system comes with pre-configured alert rules:

- **high_cost_usage**: Triggers when daily cost exceeds 80%
- **critical_cost_usage**: Triggers when daily cost exceeds 95%
- **low_quality_rate**: Triggers when validation rate falls below 70%
- **high_error_rate**: Triggers when error rate exceeds 20%
- **low_efficiency**: Triggers when efficiency drops below 0.5 results/minute
- **excessive_session_duration**: Triggers for sessions running over 2 hours

#### Custom Alert Rules

```python
from monitoring.alert_manager import AlertRule, AlertSeverity, AlertChannel

# Create custom alert rule
custom_rule = AlertRule(
    name="high_linkedin_cost",
    description="LinkedIn scraper cost exceeds budget",
    condition=lambda data: data.get('linkedin_premium_cost', 0) > 300,
    severity=AlertSeverity.HIGH,
    channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
    cooldown_seconds=1800  # 30 minutes
)

alert_manager.add_rule(custom_rule)
```

#### Notification Channels

Configure multiple notification channels:

```python
from monitoring.alert_manager import AlertManagerConfig

config = AlertManagerConfig(
    # Email configuration
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    smtp_username="your_email@gmail.com",
    smtp_password="your_password",
    email_recipients=["admin@company.com", "ops@company.com"],
    
    # Webhook configuration
    webhook_urls=["https://hooks.slack.com/your-webhook"],
    
    # Alert settings
    max_alerts_per_hour=20,
    alert_retention_hours=168  # 1 week
)

alert_manager = create_alert_manager(config)
```

#### Alert Management

```python
# Check alerts manually
monitoring_data = monitor.get_real_time_status()
triggered_alerts = alert_manager.check_alerts(monitoring_data)

# Get alert summary
alert_summary = alert_manager.get_alert_summary(hours=24)
print(f"Total alerts: {alert_summary['total_alerts']}")
print(f"Critical alerts: {alert_summary['by_severity'].get('critical', 0)}")

# Acknowledge and resolve alerts
alert_manager.acknowledge_alert(alert_index=0)
alert_manager.resolve_alert(alert_index=0)
```

## Dashboard Visualization

### ApifyDashboard Features

The dashboard system provides multiple output formats and auto-refresh capabilities:

#### Dashboard Generation

```python
# Generate comprehensive dashboard data
dashboard_data = dashboard.generate_dashboard_data()

# Key sections include:
# - key_metrics: Success rate, cost efficiency, quality scores
# - monitoring_status: Real-time system status
# - performance: 24-hour and 1-hour summaries
# - alerts: Recent alert summary
# - system_health: Overall health assessment
```

#### Multiple Output Formats

```python
# JSON format (for APIs and integrations)
json_dashboard = dashboard.export_dashboard("json")

# HTML format (for web viewing)
html_dashboard = dashboard.generate_html_dashboard()

# Text format (for console/email)
text_dashboard = dashboard.generate_text_dashboard()
```

#### Auto-Reporting

```python
from pathlib import Path
from dashboard.apify_dashboard import DashboardConfig

# Configure auto-reporting
config = DashboardConfig(
    output_format="html",
    auto_refresh_seconds=300,  # 5 minutes
    include_detailed_metrics=True,
    include_performance_charts=True
)

dashboard = create_apify_dashboard(monitor, alert_manager, config)

# Start auto-reporting
output_dir = Path("./dashboard_reports")
output_dir.mkdir(exist_ok=True)
dashboard.start_auto_reporting(output_dir, formats=["html", "json"])
```

#### System Health Assessment

The dashboard automatically assesses system health:

```python
dashboard_data = dashboard.generate_dashboard_data()
health = dashboard_data['system_health']

print(f"Health Score: {health['score']}/100")
print(f"Status: {health['status']}")  # excellent, good, warning, critical
print(f"Factors affecting health:")
for factor, value in health['factors'].items():
    print(f"  {factor}: {value}")
```

## Performance Testing

### Benchmark Suite

The performance testing framework provides comprehensive benchmarking:

```python
from benchmarks.performance_tests import run_performance_tests
from pathlib import Path

# Run complete performance test suite
output_dir = Path("./performance_results")
output_dir.mkdir(exist_ok=True)

report = run_performance_tests(
    target_operations=1000,
    concurrent_workers=5,
    output_dir=output_dir
)

print(f"Performance Grade: {report['summary']['performance_grade']}")
print(f"Average Throughput: {report['summary']['average_throughput_ops_per_second']:.2f} ops/sec")
```

### Available Benchmarks

1. **Monitoring Throughput**: Tests monitoring system throughput
2. **Alerting Latency**: Tests alert processing latency
3. **Dashboard Generation**: Tests dashboard generation performance
4. **Concurrent Load**: Tests concurrent operation handling
5. **Memory Stress**: Tests memory usage under stress

### Custom Benchmarks

```python
from benchmarks.performance_tests import PerformanceBenchmark, BenchmarkType, BenchmarkConfig

class CustomBenchmark(PerformanceBenchmark):
    def __init__(self):
        super().__init__("Custom Test", BenchmarkType.LATENCY)
    
    def run(self, config: BenchmarkConfig):
        # Implement custom benchmark logic
        pass

# Add to test suite
from benchmarks.performance_tests import PerformanceTestSuite

suite = PerformanceTestSuite()
suite.benchmarks.append(CustomBenchmark())
suite.run_all_benchmarks()
```

## Integration Examples

### Complete Workflow Integration

```python
import time
from pathlib import Path
from utils.cost_manager import ScraperType

def automated_scraping_workflow():
    """Example of complete automated scraping workflow."""
    
    # Initialize system
    cost_manager = CostManager()
    monitor = create_apify_monitor(cost_manager)
    alert_manager = create_alert_manager()
    dashboard = create_apify_dashboard(monitor, alert_manager)
    
    # Configure alert notifications
    monitor.add_event_listener(alert_manager.process_monitoring_event)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate scraping operations
        scrapers = [
            (ScraperType.GOOGLE_PLACES, {"query": "restaurants Paris"}),
            (ScraperType.LINKEDIN_PREMIUM, {"company": "TechCorp"}),
            (ScraperType.GOOGLE_MAPS_CONTACTS, {"location": "New York"})
        ]
        
        for i, (scraper_type, params) in enumerate(scrapers):
            session_id = f"workflow_session_{i}"
            
            # Start scraper
            monitor.log_scraper_start(scraper_type, session_id, params)
            
            # Simulate processing time
            time.sleep(2)
            
            # Complete scraper
            results = {
                "status": "success",
                "results": [f"result_{j}" for j in range(10)],
                "estimated_cost": 50.0
            }
            monitor.log_scraper_complete(session_id, results)
        
        # Check for alerts
        monitoring_data = monitor.get_real_time_status()
        alerts = alert_manager.check_alerts(monitoring_data)
        
        if alerts:
            print(f"Generated {len(alerts)} alerts")
        
        # Generate dashboard
        dashboard_html = dashboard.generate_html_dashboard()
        
        # Save dashboard
        output_path = Path("workflow_dashboard.html")
        output_path.write_text(dashboard_html)
        print(f"Dashboard saved to {output_path}")
        
        return monitoring_data
        
    finally:
        monitor.stop_monitoring()

# Run the workflow
result = automated_scraping_workflow()
```

### CI/CD Integration

```python
def ci_cd_performance_check():
    """Performance check for CI/CD pipeline."""
    from benchmarks.performance_tests import PerformanceTestSuite, BenchmarkConfig
    
    # Lightweight config for CI/CD
    config = BenchmarkConfig(
        target_operations=100,  # Reduced for CI
        concurrent_workers=2,
        timeout_seconds=60
    )
    
    suite = PerformanceTestSuite(config)
    results = suite.run_all_benchmarks()
    
    # Check performance criteria
    avg_success_rate = sum(r.success_rate for r in results) / len(results)
    
    if avg_success_rate < 0.95:
        raise Exception(f"Performance test failed: success rate {avg_success_rate:.1%} < 95%")
    
    print("✅ Performance tests passed")
    return True
```

## Configuration Guide

### Environment Variables

```bash
# Optional environment variables
export APIFY_MONITOR_DEBUG=true
export APIFY_ALERT_EMAIL_SMTP=smtp.gmail.com
export APIFY_ALERT_EMAIL_USER=your_email@gmail.com
export APIFY_ALERT_EMAIL_PASS=your_password
export APIFY_DASHBOARD_AUTO_REFRESH=300
export APIFY_PERFORMANCE_LOG_LEVEL=INFO
```

### Configuration Files

Create a configuration file `apify_automation_config.yaml`:

```yaml
monitoring:
  auto_start: true
  event_buffer_size: 10000
  performance_history_size: 1000
  quality_check_enabled: true

alerting:
  max_alerts_per_hour: 20
  alert_retention_hours: 168
  default_cooldown_seconds: 300
  
  email:
    smtp_server: smtp.gmail.com
    smtp_port: 587
    recipients:
      - admin@company.com
      - ops@company.com
  
  webhook:
    urls:
      - https://hooks.slack.com/your-webhook

dashboard:
  auto_refresh_seconds: 300
  output_formats: [json, html]
  include_detailed_metrics: true
  include_performance_charts: true

performance:
  benchmark_target_operations: 1000
  benchmark_concurrent_workers: 5
  benchmark_timeout_seconds: 300
```

Load configuration:

```python
import yaml
from pathlib import Path

def load_config():
    config_path = Path("apify_automation_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

**Problem**: Memory usage grows over time during monitoring.

**Solution**:
```python
# Enable garbage collection monitoring
import gc

# Force periodic garbage collection
if session_count % 100 == 0:
    gc.collect()

# Check memory limits in configuration
config = BenchmarkConfig(memory_monitoring_interval=0.5)
```

#### 2. Alert Storm

**Problem**: Too many alerts being generated.

**Solution**:
```python
# Increase cooldown periods
rule.cooldown_seconds = 3600  # 1 hour

# Adjust thresholds
rule.condition = lambda data: data.get('cost_percentage', 0) > 0.90  # 90% instead of 80%

# Enable rate limiting
config = AlertManagerConfig(max_alerts_per_hour=10)
```

#### 3. Dashboard Performance

**Problem**: Dashboard generation is slow.

**Solution**:
```python
# Disable detailed metrics for faster generation
config = DashboardConfig(
    include_detailed_metrics=False,
    include_performance_charts=False
)

# Use text format for faster generation
text_dashboard = dashboard.generate_text_dashboard()
```

#### 4. Missing Quality Data

**Problem**: Quality metrics not appearing in dashboard.

**Solution**:
```python
# Ensure quality controllers are properly initialized
from utils.quality_controller import GoogleMapsQualityController

# Check if quality results are being passed
results = {
    "results": extraction_results,  # Ensure this contains actual data
    "status": "success"
}
monitor.log_scraper_complete(session_id, results)
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed benchmark logging
config = BenchmarkConfig(enable_detailed_logging=True)
```

### Performance Monitoring

Monitor system performance during operation:

```python
import psutil

def monitor_system_resources():
    process = psutil.Process()
    
    print(f"CPU Usage: {process.cpu_percent()}%")
    print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Open Files: {len(process.open_files())}")
    print(f"Threads: {process.num_threads()}")

# Call periodically during operation
monitor_system_resources()
```

## Best Practices

### 1. Monitoring Setup

```python
# Always use context managers for monitoring
class AutoMonitor:
    def __init__(self):
        self.monitor = create_apify_monitor()
    
    def __enter__(self):
        self.monitor.start_monitoring()
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()

# Usage
with AutoMonitor() as monitor:
    # Your scraping operations
    pass
```

### 2. Alert Configuration

```python
# Use graduated alert thresholds
alert_manager.add_rule(AlertRule(
    name="cost_warning",
    condition=lambda data: data.get('daily_cost_percentage', 0) > 0.7,
    severity=AlertSeverity.LOW,
    cooldown_seconds=3600
))

alert_manager.add_rule(AlertRule(
    name="cost_critical",
    condition=lambda data: data.get('daily_cost_percentage', 0) > 0.9,
    severity=AlertSeverity.CRITICAL,
    cooldown_seconds=900
))
```

### 3. Dashboard Optimization

```python
# Cache dashboard data for better performance
import time
from functools import lru_cache

class OptimizedDashboard:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_timeout = 60  # 1 minute cache
        self._last_cache_time = 0
        self._cached_data = None
    
    def get_cached_dashboard_data(self):
        current_time = time.time()
        if (current_time - self._last_cache_time) > self._cache_timeout:
            self._cached_data = self.generate_dashboard_data()
            self._last_cache_time = current_time
        return self._cached_data
```

### 4. Error Handling

```python
# Robust error handling for production
def safe_monitoring_operation(monitor, operation_func):
    try:
        return operation_func()
    except Exception as e:
        logger.error(f"Monitoring operation failed: {e}")
        # Don't let monitoring failures break the main workflow
        return None

# Usage
result = safe_monitoring_operation(
    monitor, 
    lambda: monitor.log_scraper_complete(session_id, results)
)
```

### 5. Resource Management

```python
# Implement proper cleanup
class ManagedAutomationSystem:
    def __init__(self):
        self.monitor = None
        self.alert_manager = None
        self.dashboard = None
    
    def start(self):
        self.monitor = create_apify_monitor()
        self.alert_manager = create_alert_manager()
        self.dashboard = create_apify_dashboard(self.monitor, self.alert_manager)
        
        self.monitor.start_monitoring()
        self.monitor.add_event_listener(self.alert_manager.process_monitoring_event)
    
    def stop(self):
        if self.monitor:
            self.monitor.stop_monitoring()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
```

### 6. Production Deployment

```python
# Production-ready configuration
def create_production_system():
    # Use conservative settings for production
    config = AlertManagerConfig(
        max_alerts_per_hour=10,  # Prevent alert storms
        alert_retention_hours=720,  # 30 days
        default_cooldown_seconds=1800  # 30 minutes
    )
    
    dashboard_config = DashboardConfig(
        auto_refresh_seconds=600,  # 10 minutes
        include_detailed_metrics=True,
        include_performance_charts=False  # Reduce load
    )
    
    # Initialize with robust error handling
    try:
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager(config)
        dashboard = create_apify_dashboard(monitor, alert_manager, dashboard_config)
        
        return monitor, alert_manager, dashboard
        
    except Exception as e:
        logger.critical(f"Failed to initialize automation system: {e}")
        raise
```

## Conclusion

The Apify automation system provides a comprehensive solution for monitoring, alerting, and optimizing LinkedIn and Google Maps scraping operations. With its modular architecture, extensive configuration options, and production-ready features, it enables efficient and reliable automated data collection at scale.

For additional support or questions, refer to the troubleshooting section or check the existing documentation in the `docs/` directory.