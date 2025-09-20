#!/usr/bin/env python3
"""
Advanced alert management for Apify automation.

This module provides comprehensive alerting capabilities with customizable 
thresholds, notification channels, and alert correlation.
"""

import time
import logging
import smtplib
import json
from typing import Dict, List, Optional, Any, Callable, NamedTuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from utils.cost_manager import CostManager, ScraperType, CostAlert
from monitoring.apify_monitor import MonitoringEvent, MonitoringEventType

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: int = 300  # 5 minutes default
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a triggered alert."""
    timestamp: float
    rule_name: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool = False
    resolved: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AlertManagerConfig:
    """Configuration for alert manager."""
    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    
    # Alert settings
    max_alerts_per_hour: int = 20
    alert_retention_hours: int = 168  # 1 week
    enable_alert_correlation: bool = True
    default_cooldown_seconds: int = 300


class AlertManager:
    """Advanced alert management system."""
    
    def __init__(self, config: Optional[AlertManagerConfig] = None):
        self.config = config or AlertManagerConfig()
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: deque = deque(maxlen=10000)
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, float] = {}
        self.alert_rate_tracker: deque = deque(maxlen=100)
        
        # Initialize default rules
        self._setup_default_rules()
        
        logger.info("AlertManager initialized with %d rules", len(self.rules))
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        
        # Cost threshold alerts
        self.add_rule(AlertRule(
            name="high_cost_usage",
            description="Daily cost usage exceeds 80%",
            condition=lambda data: data.get('daily_cost_percentage', 0) > 0.8,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            cooldown_seconds=1800  # 30 minutes
        ))
        
        self.add_rule(AlertRule(
            name="critical_cost_usage",
            description="Daily cost usage exceeds 95%",
            condition=lambda data: data.get('daily_cost_percentage', 0) > 0.95,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            cooldown_seconds=900  # 15 minutes
        ))
        
        # Quality threshold alerts
        self.add_rule(AlertRule(
            name="low_quality_rate",
            description="Quality validation rate below 70%",
            condition=lambda data: data.get('validation_rate', 1.0) < 0.7,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            cooldown_seconds=3600  # 1 hour
        ))
        
        # Performance alerts
        self.add_rule(AlertRule(
            name="high_error_rate",
            description="Error rate exceeds 20%",
            condition=lambda data: data.get('error_rate', 0) > 0.2,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            cooldown_seconds=1800  # 30 minutes
        ))
        
        self.add_rule(AlertRule(
            name="low_efficiency",
            description="Scraper efficiency below 0.5 results/minute",
            condition=lambda data: data.get('efficiency_score', 0) < 0.5,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG],
            cooldown_seconds=3600  # 1 hour
        ))
        
        # Resource alerts
        self.add_rule(AlertRule(
            name="excessive_session_duration",
            description="Scraper session running for over 2 hours",
            condition=lambda data: data.get('session_duration', 0) > 7200,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            cooldown_seconds=1800  # 30 minutes
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled alert rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled alert rule: {rule_name}")
    
    def check_alerts(self, monitoring_data: Dict[str, Any]):
        """Check all alert rules against monitoring data."""
        current_time = time.time()
        
        # Rate limiting check
        if self._is_rate_limited():
            logger.warning("Alert rate limit exceeded, skipping alert check")
            return
        
        triggered_alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.cooldown_tracker:
                if current_time - self.cooldown_tracker[rule_name] < rule.cooldown_seconds:
                    continue
            
            try:
                # Prepare data for rule evaluation
                rule_data = self._prepare_rule_data(monitoring_data, rule)
                
                # Evaluate rule condition
                if rule.condition(rule_data):
                    alert = self._create_alert(rule, rule_data, current_time)
                    triggered_alerts.append(alert)
                    self.cooldown_tracker[rule_name] = current_time
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        # Process triggered alerts
        for alert in triggered_alerts:
            self._process_alert(alert)
        
        return triggered_alerts
    
    def process_monitoring_event(self, event: MonitoringEvent):
        """Process a monitoring event for potential alerts."""
        if event.event_type == MonitoringEventType.COST_ALERT:
            # Convert cost alert to our alert format
            alert = Alert(
                timestamp=event.timestamp,
                rule_name="cost_threshold",
                severity=AlertSeverity.HIGH if "CRITICAL" in event.data.get('alert_type', '') else AlertSeverity.MEDIUM,
                message=event.data.get('message', 'Cost threshold exceeded'),
                details=event.data,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL]
            )
            self._process_alert(alert)
        
        elif event.event_type == MonitoringEventType.SCRAPER_ERROR:
            # Create error alert
            alert = Alert(
                timestamp=event.timestamp,
                rule_name="scraper_error",
                severity=AlertSeverity.MEDIUM,
                message=f"Scraper error: {event.data.get('error_message', 'Unknown error')}",
                details=event.data,
                channels=[AlertChannel.LOG]
            )
            self._process_alert(alert)
    
    def _prepare_rule_data(self, monitoring_data: Dict[str, Any], rule: AlertRule) -> Dict[str, Any]:
        """Prepare data for rule evaluation."""
        rule_data = monitoring_data.copy()
        
        # Add computed metrics
        overall_stats = monitoring_data.get('overall_stats', {})
        
        # Calculate daily cost percentage
        total_cost = overall_stats.get('total_cost', 0)
        daily_budget = 1000.0  # TODO: Get from cost manager
        rule_data['daily_cost_percentage'] = total_cost / daily_budget if daily_budget > 0 else 0
        
        # Calculate error rate
        total_ops = overall_stats.get('total_operations', 0)
        success_rate = overall_stats.get('success_rate', 1.0)
        rule_data['error_rate'] = 1.0 - success_rate if total_ops > 0 else 0
        
        # Add quality metrics
        quality_metrics = monitoring_data.get('quality_metrics', {})
        rule_data.update(quality_metrics)
        
        # Add scraper-specific metrics
        scraper_stats = monitoring_data.get('scraper_stats', {})
        for scraper_type, stats in scraper_stats.items():
            rule_data[f'{scraper_type}_efficiency'] = stats.get('efficiency_score', 0)
            rule_data[f'{scraper_type}_cost'] = stats.get('total_cost', 0)
        
        return rule_data
    
    def _create_alert(self, rule: AlertRule, rule_data: Dict[str, Any], timestamp: float) -> Alert:
        """Create an alert from a triggered rule."""
        # Generate detailed message
        message = f"{rule.description}"
        
        # Add relevant metrics to message
        if "cost" in rule.name and 'daily_cost_percentage' in rule_data:
            percentage = rule_data['daily_cost_percentage'] * 100
            message += f" (Current: {percentage:.1f}%)"
        
        if "quality" in rule.name and 'validation_rate' in rule_data:
            rate = rule_data['validation_rate'] * 100
            message += f" (Current: {rate:.1f}%)"
        
        if "error" in rule.name and 'error_rate' in rule_data:
            rate = rule_data['error_rate'] * 100
            message += f" (Current: {rate:.1f}%)"
        
        return Alert(
            timestamp=timestamp,
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            details=rule_data,
            channels=rule.channels.copy()
        )
    
    def _process_alert(self, alert: Alert):
        """Process a triggered alert."""
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.alert_rate_tracker.append(time.time())
        
        logger.info(f"Alert triggered: {alert.rule_name} - {alert.message}")
        
        # Send notifications
        for channel in alert.channels:
            try:
                self._send_notification(alert, channel)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
        
        # Clean up old alerts
        self._cleanup_old_alerts()
    
    def _send_notification(self, alert: Alert, channel: AlertChannel):
        """Send alert notification via specified channel."""
        if channel == AlertChannel.LOG:
            log_level = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }[alert.severity]
            
            logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        elif channel == AlertChannel.CONSOLE:
            severity_prefix = {
                AlertSeverity.LOW: "🔵",
                AlertSeverity.MEDIUM: "🟡", 
                AlertSeverity.HIGH: "🟠",
                AlertSeverity.CRITICAL: "🔴"
            }[alert.severity]
            
            print(f"{severity_prefix} ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        elif channel == AlertChannel.EMAIL:
            if self.config.email_recipients and self.config.smtp_server:
                self._send_email_alert(alert)
        
        elif channel == AlertChannel.WEBHOOK:
            if self.config.webhook_urls:
                self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        if not self.config.smtp_server or not self.config.email_recipients:
            logger.warning("Email configuration incomplete, skipping email alert")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username or "apify-monitor@localhost"
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = f"Apify Alert [{alert.severity.value.upper()}]: {alert.rule_name}"
            
            body = f"""
Alert Details:
- Rule: {alert.rule_name}
- Severity: {alert.severity.value.upper()}
- Message: {alert.message}
- Timestamp: {datetime.fromtimestamp(alert.timestamp)}

Additional Details:
{json.dumps(alert.details, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            if self.config.smtp_username and self.config.smtp_password:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook."""
        import httpx
        
        payload = {
            'timestamp': alert.timestamp,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value,
            'message': alert.message,
            'details': alert.details
        }
        
        for webhook_url in self.config.webhook_urls:
            try:
                with httpx.Client() as client:
                    response = client.post(webhook_url, json=payload, timeout=10)
                    response.raise_for_status()
                    logger.info(f"Webhook alert sent to {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
    
    def _is_rate_limited(self) -> bool:
        """Check if alert rate limit is exceeded."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Remove old entries
        while self.alert_rate_tracker and self.alert_rate_tracker[0] < hour_ago:
            self.alert_rate_tracker.popleft()
        
        return len(self.alert_rate_tracker) >= self.config.max_alerts_per_hour
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts from history."""
        cutoff_time = time.time() - (self.config.alert_retention_hours * 3600)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff_time]
        
        if not recent_alerts:
            return {
                'period_hours': hours,
                'total_alerts': 0,
                'by_severity': {},
                'by_rule': {},
                'recent_alerts': []
            }
        
        # Group by severity
        by_severity = defaultdict(int)
        for alert in recent_alerts:
            by_severity[alert.severity.value] += 1
        
        # Group by rule
        by_rule = defaultdict(int)
        for alert in recent_alerts:
            by_rule[alert.rule_name] += 1
        
        # Get most recent alerts
        recent_alerts_sorted = sorted(recent_alerts, key=lambda a: a.timestamp, reverse=True)[:10]
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'by_severity': dict(by_severity),
            'by_rule': dict(by_rule),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                } for alert in recent_alerts_sorted
            ]
        }
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert."""
        if 0 <= alert_index < len(self.alert_history):
            self.alert_history[alert_index].acknowledged = True
            logger.info(f"Alert acknowledged: {self.alert_history[alert_index].rule_name}")
            return True
        return False
    
    def resolve_alert(self, alert_index: int) -> bool:
        """Mark an alert as resolved."""
        if 0 <= alert_index < len(self.alert_history):
            self.alert_history[alert_index].resolved = True
            logger.info(f"Alert resolved: {self.alert_history[alert_index].rule_name}")
            return True
        return False


def create_alert_manager(config: Optional[AlertManagerConfig] = None) -> AlertManager:
    """Factory function to create AlertManager instance."""
    return AlertManager(config)