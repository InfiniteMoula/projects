#!/usr/bin/env python3
"""
Enhanced progress tracking for complex data processing pipelines.

This module provides comprehensive progress tracking with real-time updates,
performance metrics, and integration with existing state management.
"""

import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging

from utils.state import SequentialRunState

logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """Comprehensive progress metrics."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    start_time: float = 0
    end_time: float = 0
    current_phase: str = "idle"
    current_item: str = ""
    
    # Performance metrics
    items_per_second: float = 0
    estimated_time_remaining: float = 0
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0
    
    # Quality metrics
    success_rate: float = 0
    error_rate: float = 0
    retry_count: int = 0
    
    # Phase-specific metrics
    phase_metrics: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.phase_metrics is None:
            self.phase_metrics = {}


@dataclass
class PhaseInfo:
    """Information about processing phases."""
    name: str
    description: str
    total_items: int = 0
    completed_items: int = 0
    start_time: float = 0
    end_time: float = 0
    status: str = "pending"  # pending, running, completed, failed


class ProgressTracker:
    """
    Enhanced progress tracker with real-time monitoring and metrics.
    
    Provides comprehensive tracking of processing progress including:
    - Multi-phase processing support
    - Real-time performance metrics
    - Memory and CPU monitoring
    - Progress persistence and recovery
    - Callback support for UI updates
    """
    
    def __init__(
        self,
        output_dir: str,
        enable_monitoring: bool = True,
        monitoring_interval: float = 1.0,
        persist_state: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.persist_state = persist_state
        
        # Progress state
        self.metrics = ProgressMetrics()
        self.phases: Dict[str, PhaseInfo] = {}
        self.callbacks: List[Callable[[ProgressMetrics], None]] = []
        
        # Monitoring
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # State persistence
        if self.persist_state:
            self.state_file = self.output_dir / "progress_state.json"
            self.sequential_state = SequentialRunState(self.output_dir / "sequential_state.json")
            self._load_state()
        
        # Performance tracking
        self._last_update_time = time.time()
        self._last_completed_count = 0
        
    def add_callback(self, callback: Callable[[ProgressMetrics], None]):
        """Add a callback function for progress updates."""
        self.callbacks.append(callback)
    
    def define_phases(self, phases: List[Dict[str, Any]]):
        """Define processing phases."""
        with self._lock:
            for phase_info in phases:
                phase = PhaseInfo(
                    name=phase_info["name"],
                    description=phase_info.get("description", ""),
                    total_items=phase_info.get("total_items", 0)
                )
                self.phases[phase.name] = phase
                
                # Initialize phase metrics
                self.metrics.phase_metrics[phase.name] = {
                    "items_processed": 0,
                    "errors": 0,
                    "avg_processing_time": 0,
                    "throughput": 0
                }
    
    def start_processing(self, total_items: int, phase: str = "main"):
        """Start processing with specified total items."""
        with self._lock:
            self.metrics.total_items = total_items
            self.metrics.completed_items = 0
            self.metrics.failed_items = 0
            self.metrics.skipped_items = 0
            self.metrics.start_time = time.time()
            self.metrics.current_phase = phase
            
            # Update phase info
            if phase in self.phases:
                self.phases[phase].status = "running"
                self.phases[phase].start_time = time.time()
                self.phases[phase].total_items = total_items
        
        # Start monitoring
        if self.enable_monitoring:
            self._start_monitoring()
        
        logger.info(f"Started processing {total_items} items in phase '{phase}'")
        self._notify_callbacks()
    
    def start_phase(self, phase_name: str, total_items: Optional[int] = None):
        """Start a specific processing phase."""
        with self._lock:
            if phase_name not in self.phases:
                self.phases[phase_name] = PhaseInfo(
                    name=phase_name,
                    description=f"Phase: {phase_name}",
                    total_items=total_items or 0
                )
                self.metrics.phase_metrics[phase_name] = {
                    "items_processed": 0,
                    "errors": 0,
                    "avg_processing_time": 0,
                    "throughput": 0
                }
            
            self.metrics.current_phase = phase_name
            self.phases[phase_name].status = "running"
            self.phases[phase_name].start_time = time.time()
            
            if total_items is not None:
                self.phases[phase_name].total_items = total_items
        
        logger.info(f"Started phase '{phase_name}' with {total_items} items")
        self._notify_callbacks()
    
    def complete_phase(self, phase_name: str):
        """Complete a processing phase."""
        with self._lock:
            if phase_name in self.phases:
                self.phases[phase_name].status = "completed"
                self.phases[phase_name].end_time = time.time()
        
        logger.info(f"Completed phase '{phase_name}'")
        self._notify_callbacks()
    
    def update_progress(
        self,
        completed_delta: int = 1,
        failed_delta: int = 0,
        skipped_delta: int = 0,
        current_item: str = "",
        extra_data: Optional[Dict] = None
    ):
        """Update progress counters."""
        with self._lock:
            self.metrics.completed_items += completed_delta
            self.metrics.failed_items += failed_delta
            self.metrics.skipped_items += skipped_delta
            
            if current_item:
                self.metrics.current_item = current_item
            
            # Update phase metrics
            current_phase = self.metrics.current_phase
            if current_phase in self.phases:
                self.phases[current_phase].completed_items += completed_delta
                
                # Update phase-specific metrics
                phase_metrics = self.metrics.phase_metrics.get(current_phase, {})
                phase_metrics["items_processed"] = phase_metrics.get("items_processed", 0) + completed_delta
                phase_metrics["errors"] = phase_metrics.get("errors", 0) + failed_delta
            
            # Calculate derived metrics
            self._update_derived_metrics()
        
        # Persist state if enabled
        if self.persist_state and completed_delta > 0:
            self.sequential_state.mark_completed(current_item, extra=extra_data)
        elif self.persist_state and failed_delta > 0:
            self.sequential_state.mark_failed(current_item, extra_data.get("error", "Unknown error") if extra_data else "Unknown error")
        
        self._notify_callbacks()
    
    def report_error(self, error_message: str, item: str = "", retry: bool = False):
        """Report an error during processing."""
        with self._lock:
            if not retry:
                self.metrics.failed_items += 1
            else:
                self.metrics.retry_count += 1
            
            # Update phase error count
            current_phase = self.metrics.current_phase
            if current_phase in self.metrics.phase_metrics:
                self.metrics.phase_metrics[current_phase]["errors"] += 1
        
        logger.error(f"Processing error in phase '{self.metrics.current_phase}': {error_message}")
        
        if self.persist_state and not retry:
            self.sequential_state.mark_failed(item, error_message)
        
        self._notify_callbacks()
    
    def complete_processing(self):
        """Mark processing as complete."""
        with self._lock:
            self.metrics.end_time = time.time()
            self.metrics.current_phase = "completed"
            
            # Complete current phase
            for phase_name, phase in self.phases.items():
                if phase.status == "running":
                    phase.status = "completed"
                    phase.end_time = time.time()
        
        # Stop monitoring
        if self.enable_monitoring:
            self._stop_monitoring()
        
        # Final metrics calculation
        self._update_derived_metrics()
        
        # Persist final state
        if self.persist_state:
            self._save_state()
        
        logger.info("Processing completed")
        self._notify_callbacks()
    
    def _update_derived_metrics(self):
        """Update calculated metrics."""
        current_time = time.time()
        
        # Calculate rates
        if self.metrics.total_items > 0:
            self.metrics.success_rate = self.metrics.completed_items / self.metrics.total_items
            self.metrics.error_rate = self.metrics.failed_items / self.metrics.total_items
        
        # Calculate throughput
        elapsed_time = current_time - self.metrics.start_time
        if elapsed_time > 0:
            self.metrics.items_per_second = self.metrics.completed_items / elapsed_time
            
            # Estimate remaining time
            remaining_items = self.metrics.total_items - self.metrics.completed_items
            if self.metrics.items_per_second > 0:
                self.metrics.estimated_time_remaining = remaining_items / self.metrics.items_per_second
        
        # Update phase throughput
        current_phase = self.metrics.current_phase
        if current_phase in self.phases and current_phase in self.metrics.phase_metrics:
            phase = self.phases[current_phase]
            phase_elapsed = current_time - phase.start_time if phase.start_time > 0 else 1
            
            if phase_elapsed > 0:
                self.metrics.phase_metrics[current_phase]["throughput"] = phase.completed_items / phase_elapsed
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            logger.warning("psutil not available, system monitoring disabled")
            process = None
        
        while self._monitoring:
            try:
                with self._lock:
                    # Update system metrics if available
                    if process:
                        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                        self.metrics.cpu_usage_percent = process.cpu_percent()
                    
                    # Update derived metrics
                    self._update_derived_metrics()
                
                # Persist state periodically
                if self.persist_state:
                    self._save_state()
                
                # Notify callbacks
                self._notify_callbacks()
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _save_state(self):
        """Save progress state to file."""
        if not self.persist_state:
            return
        
        try:
            state_data = {
                "metrics": asdict(self.metrics),
                "phases": {name: asdict(phase) for name, phase in self.phases.items()},
                "timestamp": time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save progress state: {e}")
    
    def _load_state(self):
        """Load progress state from file."""
        if not self.persist_state or not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore metrics (but don't restore transient fields)
            saved_metrics = state_data.get("metrics", {})
            if saved_metrics:
                # Only restore persistent metrics
                self.metrics.total_items = saved_metrics.get("total_items", 0)
                self.metrics.completed_items = saved_metrics.get("completed_items", 0)
                self.metrics.failed_items = saved_metrics.get("failed_items", 0)
                self.metrics.phase_metrics = saved_metrics.get("phase_metrics", {})
            
            # Restore phases
            saved_phases = state_data.get("phases", {})
            for name, phase_data in saved_phases.items():
                self.phases[name] = PhaseInfo(**phase_data)
            
            logger.info("Loaded previous progress state")
            
        except Exception as e:
            logger.error(f"Failed to load progress state: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        with self._lock:
            summary = {
                "overall": asdict(self.metrics),
                "phases": {name: asdict(phase) for name, phase in self.phases.items()},
                "completion_percentage": (self.metrics.completed_items / max(self.metrics.total_items, 1)) * 100,
                "has_errors": self.metrics.failed_items > 0,
                "is_running": self.metrics.current_phase not in ["idle", "completed"]
            }
            
            # Add time estimates
            if self.metrics.start_time > 0:
                elapsed = time.time() - self.metrics.start_time
                summary["elapsed_time"] = elapsed
                
                if self.metrics.end_time > 0:
                    summary["total_time"] = self.metrics.end_time - self.metrics.start_time
        
        return summary
    
    def get_phase_summary(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific phase."""
        with self._lock:
            if phase_name not in self.phases:
                return None
            
            phase = self.phases[phase_name]
            phase_metrics = self.metrics.phase_metrics.get(phase_name, {})
            
            return {
                "phase_info": asdict(phase),
                "metrics": phase_metrics,
                "completion_percentage": (phase.completed_items / max(phase.total_items, 1)) * 100,
                "elapsed_time": (phase.end_time or time.time()) - phase.start_time if phase.start_time > 0 else 0
            }
    
    def reset(self):
        """Reset all progress tracking."""
        with self._lock:
            self.metrics = ProgressMetrics()
            self.phases.clear()
            
        if self.persist_state and self.state_file.exists():
            self.state_file.unlink()
        
        logger.info("Progress tracking reset")


def create_progress_tracker(
    output_dir: str,
    config: Optional[Dict] = None
) -> ProgressTracker:
    """
    Factory function to create a configured ProgressTracker.
    
    Args:
        output_dir: Directory for state persistence
        config: Optional configuration dictionary
        
    Returns:
        Configured ProgressTracker instance
    """
    if config:
        enable_monitoring = config.get("enable_monitoring", True)
        monitoring_interval = config.get("monitoring_interval", 1.0)
        persist_state = config.get("persist_state", True)
    else:
        enable_monitoring = True
        monitoring_interval = 1.0
        persist_state = True
    
    return ProgressTracker(
        output_dir=output_dir,
        enable_monitoring=enable_monitoring,
        monitoring_interval=monitoring_interval,
        persist_state=persist_state
    )


if __name__ == "__main__":
    # Example usage
    import time
    import random
    
    def example_callback(metrics: ProgressMetrics):
        """Example progress callback."""
        print(f"Progress: {metrics.completed_items}/{metrics.total_items} "
              f"({metrics.completion_percentage:.1f}%) - "
              f"{metrics.items_per_second:.1f} items/sec")
    
    # Create tracker
    tracker = create_progress_tracker("/tmp/progress_test")
    tracker.add_callback(example_callback)
    
    # Define phases
    tracker.define_phases([
        {"name": "phase1", "description": "First phase", "total_items": 50},
        {"name": "phase2", "description": "Second phase", "total_items": 30}
    ])
    
    # Simulate processing
    tracker.start_phase("phase1", 50)
    
    for i in range(50):
        time.sleep(0.1)  # Simulate work
        tracker.update_progress(1, current_item=f"item_{i}")
        
        # Simulate some failures
        if random.random() < 0.1:
            tracker.report_error(f"Error processing item_{i}", f"item_{i}")
    
    tracker.complete_phase("phase1")
    
    # Second phase
    tracker.start_phase("phase2", 30)
    
    for i in range(30):
        time.sleep(0.05)
        tracker.update_progress(1, current_item=f"phase2_item_{i}")
    
    tracker.complete_phase("phase2")
    tracker.complete_processing()
    
    # Print summary
    print(json.dumps(tracker.get_summary(), indent=2, default=str))