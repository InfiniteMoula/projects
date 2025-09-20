#!/usr/bin/env python3
"""
Performance dashboard for Apify automation monitoring.

This module provides comprehensive dashboards for visualizing monitoring data,
performance metrics, and alert status.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from monitoring.apify_monitor import ApifyMonitor
from monitoring.alert_manager import AlertManager, AlertSeverity
from utils.cost_manager import CostManager, ScraperType

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    output_format: str = "json"  # json, html, text
    auto_refresh_seconds: int = 60
    include_detailed_metrics: bool = True
    include_performance_charts: bool = True
    export_path: Optional[Path] = None


class ApifyDashboard:
    """Performance dashboard for Apify automation."""
    
    def __init__(
        self,
        monitor: ApifyMonitor,
        alert_manager: AlertManager,
        config: Optional[DashboardConfig] = None
    ):
        self.monitor = monitor
        self.alert_manager = alert_manager
        self.config = config or DashboardConfig()
        
        logger.info("ApifyDashboard initialized")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        current_time = time.time()
        
        # Get monitoring status
        monitoring_status = self.monitor.get_real_time_status()
        
        # Get performance summary
        performance_24h = self.monitor.get_performance_summary(hours=24)
        performance_1h = self.monitor.get_performance_summary(hours=1)
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary(hours=24)
        
        # Calculate key metrics
        key_metrics = self._calculate_key_metrics(monitoring_status, performance_24h)
        
        dashboard_data = {
            'timestamp': current_time,
            'generation_time': datetime.fromtimestamp(current_time).isoformat(),
            'uptime_seconds': monitoring_status.get('uptime_seconds', 0),
            'key_metrics': key_metrics,
            'monitoring_status': monitoring_status,
            'performance': {
                'last_24_hours': performance_24h,
                'last_hour': performance_1h
            },
            'alerts': alert_summary,
            'system_health': self._assess_system_health(monitoring_status, alert_summary)
        }
        
        if self.config.include_detailed_metrics:
            dashboard_data['detailed_metrics'] = self._generate_detailed_metrics()
        
        if self.config.include_performance_charts:
            dashboard_data['chart_data'] = self._generate_chart_data()
        
        return dashboard_data
    
    def _calculate_key_metrics(self, monitoring_status: Dict, performance_data: Dict) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        overall_stats = monitoring_status.get('overall_stats', {})
        
        # Success rate
        success_rate = overall_stats.get('success_rate', 0) * 100
        
        # Cost efficiency (results per credit)
        total_cost = performance_data.get('total_cost', 0)
        total_results = performance_data.get('total_results', 0)
        cost_per_result = total_cost / total_results if total_results > 0 else 0
        
        # Average efficiency (results per minute)
        avg_efficiency = performance_data.get('average_efficiency', 0)
        
        # Quality score
        quality_metrics = monitoring_status.get('quality_metrics', {})
        avg_quality_score = quality_metrics.get('average_score', 0)
        
        # Alert status
        recent_alerts = len(monitoring_status.get('recent_alerts', []))
        
        return {
            'success_rate_percent': round(success_rate, 1),
            'cost_per_result': round(cost_per_result, 2),
            'results_per_minute': round(avg_efficiency, 2),
            'average_quality_score': round(avg_quality_score, 1),
            'active_alerts': recent_alerts,
            'total_operations_24h': performance_data.get('total_operations', 0),
            'total_cost_24h': round(performance_data.get('total_cost', 0), 2)
        }
    
    def _assess_system_health(self, monitoring_status: Dict, alert_summary: Dict) -> Dict[str, Any]:
        """Assess overall system health."""
        overall_stats = monitoring_status.get('overall_stats', {})
        
        # Health score calculation (0-100)
        health_score = 100
        
        # Deduct points for low success rate
        success_rate = overall_stats.get('success_rate', 1.0)
        if success_rate < 0.9:
            health_score -= (0.9 - success_rate) * 100
        
        # Deduct points for recent alerts
        critical_alerts = alert_summary.get('by_severity', {}).get('critical', 0)
        high_alerts = alert_summary.get('by_severity', {}).get('high', 0)
        medium_alerts = alert_summary.get('by_severity', {}).get('medium', 0)
        
        health_score -= critical_alerts * 20
        health_score -= high_alerts * 10
        health_score -= medium_alerts * 5
        
        # Ensure health score is between 0 and 100
        health_score = max(0, min(100, health_score))
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
            color = "green"
        elif health_score >= 75:
            status = "good"
            color = "yellow"
        elif health_score >= 50:
            status = "warning"
            color = "orange"
        else:
            status = "critical"
            color = "red"
        
        return {
            'score': round(health_score, 1),
            'status': status,
            'color': color,
            'factors': {
                'success_rate': success_rate,
                'critical_alerts': critical_alerts,
                'high_alerts': high_alerts,
                'medium_alerts': medium_alerts
            }
        }
    
    def _generate_detailed_metrics(self) -> Dict[str, Any]:
        """Generate detailed metrics breakdown."""
        monitoring_status = self.monitor.get_real_time_status()
        scraper_stats = monitoring_status.get('scraper_stats', {})
        
        detailed_metrics = {}
        
        for scraper_type, stats in scraper_stats.items():
            efficiency = stats.get('efficiency_score', 0)
            total_cost = stats.get('total_cost', 0)
            success_ops = stats.get('successful_operations', 0)
            failed_ops = stats.get('failed_operations', 0)
            total_ops = success_ops + failed_ops
            
            # Convert enum to string for JSON serialization
            scraper_key = scraper_type.value if hasattr(scraper_type, 'value') else str(scraper_type)
            
            detailed_metrics[scraper_key] = {
                'operations': {
                    'total': total_ops,
                    'successful': success_ops,
                    'failed': failed_ops,
                    'success_rate': (success_ops / total_ops) if total_ops > 0 else 0
                },
                'performance': {
                    'efficiency_score': efficiency,
                    'total_cost': total_cost,
                    'average_cost_per_operation': (total_cost / total_ops) if total_ops > 0 else 0,
                    'total_duration': stats.get('total_duration', 0)
                },
                'quality': {
                    'average_score': stats.get('average_quality_score', 0)
                }
            }
        
        return detailed_metrics
    
    def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for performance charts."""
        # Get recent performance history
        recent_performance = list(self.monitor.performance_history)
        
        if not recent_performance:
            return {'message': 'No performance data available for charts'}
        
        # Sort by timestamp
        recent_performance.sort(key=lambda x: x['timestamp'])
        
        # Generate time series data for different metrics
        time_series = {
            'timestamps': [],
            'cost_per_hour': [],
            'results_per_hour': [],
            'efficiency_scores': [],
            'operations_per_hour': []
        }
        
        # Group data by hour
        hourly_data = {}
        for entry in recent_performance:
            hour_key = int(entry['timestamp'] // 3600) * 3600  # Round to hour
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {
                    'cost': 0,
                    'results': 0,
                    'operations': 0,
                    'total_efficiency': 0,
                    'efficiency_count': 0
                }
            
            hourly_data[hour_key]['cost'] += entry['cost']
            hourly_data[hour_key]['results'] += entry['results_count']
            hourly_data[hour_key]['operations'] += 1
            hourly_data[hour_key]['total_efficiency'] += entry['efficiency']
            hourly_data[hour_key]['efficiency_count'] += 1
        
        # Convert to time series
        for hour_timestamp in sorted(hourly_data.keys()):
            data = hourly_data[hour_timestamp]
            
            time_series['timestamps'].append(hour_timestamp)
            time_series['cost_per_hour'].append(data['cost'])
            time_series['results_per_hour'].append(data['results'])
            time_series['operations_per_hour'].append(data['operations'])
            
            avg_efficiency = (data['total_efficiency'] / data['efficiency_count']) if data['efficiency_count'] > 0 else 0
            time_series['efficiency_scores'].append(avg_efficiency)
        
        # Generate scraper breakdown chart data
        scraper_breakdown = {}
        for entry in recent_performance:
            scraper_type = entry['scraper_type']
            if scraper_type not in scraper_breakdown:
                scraper_breakdown[scraper_type] = {
                    'total_cost': 0,
                    'total_results': 0,
                    'total_operations': 0
                }
            
            scraper_breakdown[scraper_type]['total_cost'] += entry['cost']
            scraper_breakdown[scraper_type]['total_results'] += entry['results_count']
            scraper_breakdown[scraper_type]['total_operations'] += 1
        
        return {
            'time_series': time_series,
            'scraper_breakdown': scraper_breakdown,
            'data_points': len(recent_performance),
            'time_range_hours': (max(time_series['timestamps']) - min(time_series['timestamps'])) / 3600 if time_series['timestamps'] else 0
        }
    
    def generate_text_dashboard(self) -> str:
        """Generate a text-based dashboard."""
        dashboard_data = self.generate_dashboard_data()
        
        lines = []
        lines.append("=" * 80)
        lines.append("APIFY AUTOMATION DASHBOARD")
        lines.append("=" * 80)
        lines.append(f"Generated: {dashboard_data['generation_time']}")
        lines.append(f"Uptime: {dashboard_data['uptime_seconds'] / 3600:.1f} hours")
        lines.append("")
        
        # System health
        health = dashboard_data['system_health']
        lines.append(f"SYSTEM HEALTH: {health['status'].upper()} ({health['score']}/100)")
        lines.append("-" * 40)
        
        # Key metrics
        metrics = dashboard_data['key_metrics']
        lines.append("KEY METRICS:")
        lines.append(f"  Success Rate: {metrics['success_rate_percent']}%")
        lines.append(f"  Cost per Result: {metrics['cost_per_result']} credits")
        lines.append(f"  Results per Minute: {metrics['results_per_minute']}")
        lines.append(f"  Average Quality Score: {metrics['average_quality_score']}")
        lines.append(f"  Active Alerts: {metrics['active_alerts']}")
        lines.append("")
        
        # Recent performance
        perf_24h = dashboard_data['performance']['last_24_hours']
        lines.append("24-HOUR PERFORMANCE:")
        lines.append(f"  Total Operations: {perf_24h.get('total_operations', 0)}")
        lines.append(f"  Total Cost: {perf_24h.get('total_cost', 0):.2f} credits")
        lines.append(f"  Total Results: {perf_24h.get('total_results', 0)}")
        lines.append(f"  Average Efficiency: {perf_24h.get('average_efficiency', 0):.2f} results/min")
        lines.append("")
        
        # Scraper breakdown
        scraper_breakdown = perf_24h.get('scraper_breakdown', {})
        if scraper_breakdown:
            lines.append("SCRAPER BREAKDOWN (24h):")
            for scraper_type, stats in scraper_breakdown.items():
                lines.append(f"  {scraper_type.upper()}:")
                lines.append(f"    Operations: {stats['operations']}")
                lines.append(f"    Cost: {stats['total_cost']:.2f} credits")
                lines.append(f"    Results: {stats['total_results']}")
                lines.append(f"    Efficiency: {stats['avg_efficiency']:.2f} results/min")
            lines.append("")
        
        # Recent alerts
        alerts = dashboard_data['alerts']
        if alerts['total_alerts'] > 0:
            lines.append("RECENT ALERTS (24h):")
            lines.append(f"  Total: {alerts['total_alerts']}")
            
            by_severity = alerts['by_severity']
            if by_severity:
                lines.append("  By Severity:")
                for severity, count in by_severity.items():
                    lines.append(f"    {severity.capitalize()}: {count}")
            
            recent_alerts = alerts['recent_alerts'][:5]  # Show last 5
            if recent_alerts:
                lines.append("  Latest Alerts:")
                for alert in recent_alerts:
                    timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                    lines.append(f"    [{timestamp}] {alert['severity'].upper()}: {alert['message']}")
        else:
            lines.append("RECENT ALERTS: None")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_html_dashboard(self) -> str:
        """Generate an HTML dashboard."""
        dashboard_data = self.generate_dashboard_data()
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apify Automation Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 8px;
        }
        .health-excellent { color: #27ae60; }
        .health-good { color: #f39c12; }
        .health-warning { color: #e67e22; }
        .health-critical { color: #e74c3c; }
        .section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .section h2 {
            color: #2c3e50;
            margin-top: 0;
        }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .alert-critical { background-color: #fadbd8; border-left: 4px solid #e74c3c; }
        .alert-high { background-color: #fdebd0; border-left: 4px solid #e67e22; }
        .alert-medium { background-color: #fef9e7; border-left: 4px solid #f39c12; }
        .alert-low { background-color: #ebedef; border-left: 4px solid #85929e; }
        .refresh-time {
            text-align: center;
            color: #7f8c8d;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Apify Automation Dashboard</h1>
            <p>Real-time monitoring and performance analytics</p>
            <p>Uptime: {uptime:.1f} hours | Generated: {generation_time}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value health-{health_color}">{health_score}</div>
                <div class="metric-label">System Health Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{success_rate}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{cost_per_result}</div>
                <div class="metric-label">Cost per Result</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results_per_minute}</div>
                <div class="metric-label">Results/Minute</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{quality_score}</div>
                <div class="metric-label">Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{active_alerts}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
        </div>
        
        <div class="section">
            <h2>24-Hour Performance Summary</h2>
            <p><strong>Total Operations:</strong> {total_operations}</p>
            <p><strong>Total Cost:</strong> {total_cost:.2f} credits</p>
            <p><strong>Total Results:</strong> {total_results}</p>
            <p><strong>Average Efficiency:</strong> {avg_efficiency:.2f} results/minute</p>
        </div>
        
        {alerts_section}
        
        <div class="refresh-time">
            Dashboard auto-refreshes every {refresh_seconds} seconds
        </div>
    </div>
</body>
</html>
        """.strip()
        
        # Prepare template variables
        health = dashboard_data['system_health']
        metrics = dashboard_data['key_metrics']
        perf_24h = dashboard_data['performance']['last_24_hours']
        alerts = dashboard_data['alerts']
        
        # Generate alerts section
        alerts_html = ""
        if alerts['total_alerts'] > 0:
            alerts_html = '<div class="section"><h2>Recent Alerts</h2>'
            for alert in alerts['recent_alerts'][:10]:
                severity = alert['severity']
                timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                alerts_html += f'<div class="alert alert-{severity}">[{timestamp}] <strong>{severity.upper()}:</strong> {alert["message"]}</div>'
            alerts_html += '</div>'
        else:
            alerts_html = '<div class="section"><h2>Recent Alerts</h2><p>No recent alerts</p></div>'
        
        return html_template.format(
            uptime=dashboard_data['uptime_seconds'] / 3600,
            generation_time=dashboard_data['generation_time'],
            health_score=health['score'],
            health_color=health['color'],
            success_rate=metrics['success_rate_percent'],
            cost_per_result=metrics['cost_per_result'],
            results_per_minute=metrics['results_per_minute'],
            quality_score=metrics['average_quality_score'],
            active_alerts=metrics['active_alerts'],
            total_operations=perf_24h.get('total_operations', 0),
            total_cost=perf_24h.get('total_cost', 0),
            total_results=perf_24h.get('total_results', 0),
            avg_efficiency=perf_24h.get('average_efficiency', 0),
            alerts_section=alerts_html,
            refresh_seconds=self.config.auto_refresh_seconds
        )
    
    def export_dashboard(self, format_type: str = "json", file_path: Optional[Path] = None) -> str:
        """Export dashboard in specified format."""
        if format_type == "json":
            dashboard_data = self.generate_dashboard_data()
            content = json.dumps(dashboard_data, indent=2)
        elif format_type == "html":
            content = self.generate_html_dashboard()
        elif format_type == "text":
            content = self.generate_text_dashboard()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if file_path:
            file_path.write_text(content)
            logger.info(f"Dashboard exported to {file_path}")
        
        return content
    
    def start_auto_reporting(self, output_dir: Path, formats: List[str] = None):
        """Start automated dashboard reporting."""
        if formats is None:
            formats = ["json", "html"]
        
        import threading
        import time
        
        def reporting_loop():
            while True:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    for format_type in formats:
                        filename = f"apify_dashboard_{timestamp}.{format_type}"
                        file_path = output_dir / filename
                        self.export_dashboard(format_type, file_path)
                    
                    time.sleep(self.config.auto_refresh_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in auto-reporting: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        reporting_thread = threading.Thread(target=reporting_loop, daemon=True)
        reporting_thread.start()
        logger.info(f"Auto-reporting started to {output_dir}")


def create_apify_dashboard(
    monitor: ApifyMonitor,
    alert_manager: AlertManager,
    config: Optional[DashboardConfig] = None
) -> ApifyDashboard:
    """Factory function to create ApifyDashboard instance."""
    return ApifyDashboard(monitor, alert_manager, config)