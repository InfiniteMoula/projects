#!/usr/bin/env python3
"""
Real-time monitoring for Apify automation processes.

This module provides comprehensive monitoring capabilities for Apify operations,
integrating with existing cost management and quality control systems.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta

from utils.cost_manager import CostManager, ScraperType, CostAlert
from utils.quality_controller import GoogleMapsQualityController, LinkedInQualityController

logger = logging.getLogger(__name__)


class MonitoringEventType(Enum):
    """Types of monitoring events."""
    SCRAPER_START = "scraper_start"
    SCRAPER_COMPLETE = "scraper_complete"
    SCRAPER_ERROR = "scraper_error"
    QUALITY_CHECK = "quality_check"
    COST_ALERT = "cost_alert"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    timestamp: float
    event_type: MonitoringEventType
    scraper_type: Optional[ScraperType]
    data: Dict[str, Any]
    session_id: Optional[str] = None


class MonitoringMetrics:
    """Container for real-time monitoring metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.scraper_stats: Dict[ScraperType, Dict[str, Any]] = defaultdict(lambda: {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_cost': 0.0,
            'total_duration': 0.0,
            'average_quality_score': 0.0,
            'last_operation_time': None,
            'efficiency_score': 0.0  # results per minute
        })
        self.quality_metrics: Dict[str, Any] = {
            'total_validated': 0,
            'validation_rate': 0.0,
            'average_score': 0.0,
            'field_coverage': defaultdict(float),
            'common_issues': defaultdict(int)
        }
        self.alert_history: List[Dict[str, Any]] = []
        
    def add_event(self, event: MonitoringEvent):
        """Add a monitoring event."""
        self.events.append(event)
        self._update_metrics(event)
    
    def _update_metrics(self, event: MonitoringEvent):
        """Update metrics based on event."""
        if event.scraper_type:
            stats = self.scraper_stats[event.scraper_type]
            
            if event.event_type == MonitoringEventType.SCRAPER_START:
                stats['total_operations'] += 1
                stats['last_operation_time'] = event.timestamp
                
            elif event.event_type == MonitoringEventType.SCRAPER_COMPLETE:
                stats['successful_operations'] += 1
                if 'cost' in event.data:
                    stats['total_cost'] += event.data['cost']
                if 'duration' in event.data:
                    stats['total_duration'] += event.data['duration']
                if 'quality_score' in event.data and event.data['quality_score'] is not None:
                    self._update_quality_average(stats, event.data['quality_score'])
                    
            elif event.event_type == MonitoringEventType.SCRAPER_ERROR:
                stats['failed_operations'] += 1
                
            # Calculate efficiency (results per minute)
            if stats['total_duration'] > 0:
                stats['efficiency_score'] = (stats['successful_operations'] * 60) / stats['total_duration']
        
        if event.event_type == MonitoringEventType.QUALITY_CHECK:
            self._update_quality_metrics(event.data)
            
        elif event.event_type == MonitoringEventType.COST_ALERT:
            self.alert_history.append({
                'timestamp': event.timestamp,
                'alert_type': event.data.get('alert_type'),
                'message': event.data.get('message'),
                'scraper_type': event.scraper_type.value if event.scraper_type else None
            })
    
    def _update_quality_average(self, stats: Dict, new_score: float):
        """Update rolling average quality score."""
        current_avg = stats.get('average_quality_score', 0.0)
        success_count = stats.get('successful_operations', 1)
        stats['average_quality_score'] = ((current_avg * (success_count - 1)) + new_score) / success_count
    
    def _update_quality_metrics(self, quality_data: Dict[str, Any]):
        """Update quality metrics from quality check data."""
        self.quality_metrics['total_validated'] += quality_data.get('total_records', 0)
        
        if 'validation_rate' in quality_data:
            # Rolling average of validation rate
            current_rate = self.quality_metrics['validation_rate']
            new_rate = quality_data['validation_rate']
            count = self.quality_metrics['total_validated']
            if count > 0:
                self.quality_metrics['validation_rate'] = ((current_rate * (count - 1)) + new_rate) / count
        
        if 'field_coverage' in quality_data:
            for field, coverage in quality_data['field_coverage'].items():
                self.quality_metrics['field_coverage'][field] = coverage
        
        if 'common_issues' in quality_data:
            for issue, count in quality_data['common_issues']:
                self.quality_metrics['common_issues'][issue] += count


class ApifyMonitor:
    """Real-time monitor for Apify automation processes."""
    
    def __init__(self, cost_manager: Optional[CostManager] = None):
        self.cost_manager = cost_manager or CostManager()
        self.quality_controllers = {
            'google_maps': GoogleMapsQualityController(),
            'linkedin': LinkedInQualityController()
        }
        
        self.metrics = MonitoringMetrics()
        self.event_listeners: List[Callable[[MonitoringEvent], None]] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.session_data: Dict[str, Any] = {}
        self.performance_history: deque = deque(maxlen=1000)
        
        logger.info("ApifyMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Real-time monitoring stopped")
    
    def add_event_listener(self, listener: Callable[[MonitoringEvent], None]):
        """Add an event listener for real-time monitoring."""
        self.event_listeners.append(listener)
    
    def log_scraper_start(self, scraper_type: ScraperType, session_id: str, operation_details: Dict[str, Any]):
        """Log the start of a scraper operation."""
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type=MonitoringEventType.SCRAPER_START,
            scraper_type=scraper_type,
            data=operation_details,
            session_id=session_id
        )
        self._emit_event(event)
        
        # Store session data
        self.session_data[session_id] = {
            'start_time': event.timestamp,
            'scraper_type': scraper_type,
            'details': operation_details
        }
    
    def log_scraper_complete(self, session_id: str, results: Dict[str, Any]):
        """Log completion of a scraper operation."""
        if session_id not in self.session_data:
            logger.warning(f"No session data found for {session_id}")
            return
        
        session = self.session_data[session_id]
        duration = time.time() - session['start_time']
        
        # Perform quality check if results provided
        quality_score = None
        if 'results' in results and results['results']:
            controller_key = 'google_maps' if 'google' in session['scraper_type'].value else 'linkedin'
            controller = self.quality_controllers.get(controller_key)
            if controller and hasattr(controller, 'validate_extraction_results'):
                try:
                    # Convert results to DataFrame if needed
                    import pandas as pd
                    
                    results_data = results['results']
                    if isinstance(results_data, list):
                        if all(isinstance(item, dict) for item in results_data):
                            # Convert list of dicts to DataFrame
                            df_results = pd.DataFrame(results_data)
                        else:
                            # Simple list - create basic DataFrame
                            df_results = pd.DataFrame({'result': results_data})
                    else:
                        df_results = results_data
                    
                    validated_results = controller.validate_extraction_results(
                        df_results, session['scraper_type'].value
                    )
                    quality_report = controller.generate_quality_report(validated_results)
                    quality_score = quality_report.get('summary', {}).get('average_score', 0)
                    
                    # Log quality check event
                    self._emit_event(MonitoringEvent(
                        timestamp=time.time(),
                        event_type=MonitoringEventType.QUALITY_CHECK,
                        scraper_type=session['scraper_type'],
                        data=quality_report.get('summary', {}),
                        session_id=session_id
                    ))
                except Exception as e:
                    logger.warning(f"Quality check failed for {session_id}: {e}")
                    # Fall back to basic quality assessment
                    results_count = len(results.get('results', []))
                    quality_score = min(100, results_count * 10)  # Simple scoring
        
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type=MonitoringEventType.SCRAPER_COMPLETE,
            scraper_type=session['scraper_type'],
            data={
                'duration': duration,
                'cost': results.get('estimated_cost', 0),
                'results_count': len(results.get('results', [])),
                'quality_score': quality_score,
                'success': results.get('status') == 'success'
            },
            session_id=session_id
        )
        self._emit_event(event)
        
        # Track performance
        self.performance_history.append({
            'timestamp': event.timestamp,
            'scraper_type': session['scraper_type'].value,
            'duration': duration,
            'cost': results.get('estimated_cost', 0),
            'results_count': len(results.get('results', [])),
            'efficiency': len(results.get('results', [])) / duration if duration > 0 else 0
        })
        
        # Cleanup session data
        del self.session_data[session_id]
    
    def log_scraper_error(self, session_id: str, error: Exception):
        """Log an error in scraper operation."""
        scraper_type = None
        if session_id in self.session_data:
            scraper_type = self.session_data[session_id]['scraper_type']
            del self.session_data[session_id]
        
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type=MonitoringEventType.SCRAPER_ERROR,
            scraper_type=scraper_type,
            data={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'session_id': session_id
            },
            session_id=session_id
        )
        self._emit_event(event)
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time monitoring status."""
        current_time = time.time()
        uptime = current_time - self.metrics.start_time
        
        # Get recent cost alerts
        recent_alerts = self.cost_manager.check_budget_thresholds()
        for alert in recent_alerts:
            self._emit_event(MonitoringEvent(
                timestamp=current_time,
                event_type=MonitoringEventType.COST_ALERT,
                scraper_type=alert.scraper,
                data={
                    'alert_type': alert.alert_type,
                    'message': alert.message,
                    'current_cost': alert.current_cost,
                    'budget_limit': alert.budget_limit
                }
            ))
        
        # Calculate overall metrics
        total_operations = sum(stats['total_operations'] for stats in self.metrics.scraper_stats.values())
        total_successful = sum(stats['successful_operations'] for stats in self.metrics.scraper_stats.values())
        total_cost = sum(stats['total_cost'] for stats in self.metrics.scraper_stats.values())
        
        success_rate = (total_successful / total_operations) if total_operations > 0 else 0
        
        return {
            'timestamp': current_time,
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'overall_stats': {
                'total_operations': total_operations,
                'success_rate': success_rate,
                'total_cost': total_cost,
                'active_sessions': len(self.session_data),
                'recent_alerts': len(recent_alerts)
            },
            'scraper_stats': {
                (k.value if hasattr(k, 'value') else str(k)): v 
                for k, v in self.metrics.scraper_stats.items()
            },
            'quality_metrics': dict(self.metrics.quality_metrics),
            'recent_alerts': [
                {
                    'type': alert.alert_type,
                    'message': alert.message,
                    'scraper': alert.scraper.value,
                    'timestamp': alert.timestamp
                } for alert in recent_alerts
            ]
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_performance = [p for p in self.performance_history if p['timestamp'] > cutoff_time]
        
        if not recent_performance:
            return {'message': 'No performance data available'}
        
        # Calculate aggregated metrics
        total_operations = len(recent_performance)
        total_cost = sum(p['cost'] for p in recent_performance)
        total_results = sum(p['results_count'] for p in recent_performance)
        avg_efficiency = sum(p['efficiency'] for p in recent_performance) / total_operations
        
        # Group by scraper type
        by_scraper = defaultdict(list)
        for p in recent_performance:
            by_scraper[p['scraper_type']].append(p)
        
        scraper_summary = {}
        for scraper_type, ops in by_scraper.items():
            scraper_summary[scraper_type] = {
                'operations': len(ops),
                'total_cost': sum(op['cost'] for op in ops),
                'total_results': sum(op['results_count'] for op in ops),
                'avg_duration': sum(op['duration'] for op in ops) / len(ops),
                'avg_efficiency': sum(op['efficiency'] for op in ops) / len(ops)
            }
        
        return {
            'period_hours': hours,
            'total_operations': total_operations,
            'total_cost': total_cost,
            'total_results': total_results,
            'average_efficiency': avg_efficiency,
            'cost_per_result': total_cost / total_results if total_results > 0 else 0,
            'scraper_breakdown': scraper_summary
        }
    
    def _emit_event(self, event: MonitoringEvent):
        """Emit a monitoring event to all listeners."""
        self.metrics.add_event(event)
        
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Check for cost alerts
                alerts = self.cost_manager.check_budget_thresholds()
                for alert in alerts:
                    self._emit_event(MonitoringEvent(
                        timestamp=time.time(),
                        event_type=MonitoringEventType.COST_ALERT,
                        scraper_type=alert.scraper,
                        data={
                            'alert_type': alert.alert_type,
                            'message': alert.message,
                            'current_cost': alert.current_cost,
                            'budget_limit': alert.budget_limit
                        }
                    ))
                
                # Clean up old events (keep last 24 hours)
                cutoff_time = time.time() - 86400  # 24 hours
                while self.metrics.events and self.metrics.events[0].timestamp < cutoff_time:
                    self.metrics.events.popleft()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error


def create_apify_monitor(cost_manager: Optional[CostManager] = None) -> ApifyMonitor:
    """Factory function to create ApifyMonitor instance."""
    return ApifyMonitor(cost_manager)