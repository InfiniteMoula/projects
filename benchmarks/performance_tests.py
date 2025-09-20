#!/usr/bin/env python3
"""
Performance benchmarking framework for Apify automation.

This module provides comprehensive performance testing and benchmarking
capabilities for the Apify automation system.
"""

import time
import statistics
import logging
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import concurrent.futures
from pathlib import Path
import json
import psutil
import gc

from monitoring.apify_monitor import ApifyMonitor, create_apify_monitor
from monitoring.alert_manager import AlertManager, create_alert_manager
from dashboard.apify_dashboard import ApifyDashboard, create_apify_dashboard
from utils.cost_manager import CostManager, ScraperType

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    COST_EFFICIENCY = "cost_efficiency"
    CONCURRENT_LOAD = "concurrent_load"
    STRESS_TEST = "stress_test"


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    duration_seconds: float
    operations_count: int
    success_rate: float
    average_latency_ms: float
    throughput_ops_per_second: float
    memory_usage_mb: float
    cost_efficiency_score: float
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    target_operations: int = 1000
    concurrent_workers: int = 5
    timeout_seconds: int = 300
    memory_monitoring_interval: float = 1.0
    cost_budget_limit: float = 1000.0
    enable_detailed_logging: bool = False
    output_directory: Optional[Path] = None


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType):
        self.name = name
        self.benchmark_type = benchmark_type
        self.results: List[BenchmarkResult] = []
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run the benchmark."""
        raise NotImplementedError
    
    def cleanup(self):
        """Cleanup after benchmark."""
        pass


class MonitoringThroughputBenchmark(PerformanceBenchmark):
    """Benchmark for monitoring system throughput."""
    
    def __init__(self):
        super().__init__("Monitoring Throughput", BenchmarkType.THROUGHPUT)
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run monitoring throughput benchmark."""
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        latencies = []
        errors = []
        
        try:
            # Generate operations
            for i in range(config.target_operations):
                session_id = f"throughput_test_{i}"
                
                # Measure operation latency
                op_start = time.time()
                
                try:
                    monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {
                        "query": f"test query {i}",
                        "max_results": 10
                    })
                    
                    # Simulate completion
                    monitor.log_scraper_complete(session_id, {
                        "status": "success",
                        "results": [f"result_{j}" for j in range(5)],
                        "estimated_cost": 10.0
                    })
                    
                    op_end = time.time()
                    latencies.append((op_end - op_start) * 1000)  # Convert to ms
                    
                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Error in operation {i}: {e}")
                
                # Small delay to prevent overwhelming
                if i % 100 == 0:
                    time.sleep(0.01)
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            monitor.stop_monitoring()
        
        # Calculate metrics
        duration = end_time - start_time
        success_count = len(latencies)
        success_rate = success_count / config.target_operations
        avg_latency = statistics.mean(latencies) if latencies else 0
        throughput = success_count / duration if duration > 0 else 0
        memory_delta = end_memory - start_memory
        
        # Cost efficiency (operations per credit spent)
        total_cost = success_count * 10.0  # 10 credits per operation
        cost_efficiency = success_count / total_cost if total_cost > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            duration_seconds=duration,
            operations_count=success_count,
            success_rate=success_rate,
            average_latency_ms=avg_latency,
            throughput_ops_per_second=throughput,
            memory_usage_mb=memory_delta,
            cost_efficiency_score=cost_efficiency,
            detailed_metrics={
                'latency_p50': statistics.median(latencies) if latencies else 0,
                'latency_p95': self._percentile(latencies, 95) if latencies else 0,
                'latency_p99': self._percentile(latencies, 99) if latencies else 0,
                'error_count': len(errors),
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory
            },
            error_details=errors[:10]  # Keep first 10 errors
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]


class AlertingLatencyBenchmark(PerformanceBenchmark):
    """Benchmark for alerting system latency."""
    
    def __init__(self):
        super().__init__("Alerting Latency", BenchmarkType.LATENCY)
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run alerting latency benchmark."""
        alert_manager = create_alert_manager()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        latencies = []
        errors = []
        
        try:
            for i in range(config.target_operations):
                # Create test monitoring data that should trigger alerts
                monitoring_data = {
                    'overall_stats': {
                        'total_cost': 850.0 + (i % 100),  # Varying cost near threshold
                        'total_operations': 100,
                        'success_rate': 0.5 if i % 10 == 0 else 0.9  # Occasional low success rate
                    },
                    'quality_metrics': {
                        'validation_rate': 0.6 if i % 20 == 0 else 0.8,  # Occasional low quality
                        'average_score': 70.0
                    }
                }
                
                # Measure alert checking latency
                alert_start = time.time()
                
                try:
                    triggered_alerts = alert_manager.check_alerts(monitoring_data)
                    alert_end = time.time()
                    
                    latencies.append((alert_end - alert_start) * 1000)  # Convert to ms
                    
                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Error in alert check {i}: {e}")
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
        
        # Calculate metrics
        duration = end_time - start_time
        success_count = len(latencies)
        success_rate = success_count / config.target_operations
        avg_latency = statistics.mean(latencies) if latencies else 0
        throughput = success_count / duration if duration > 0 else 0
        memory_delta = end_memory - start_memory
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            duration_seconds=duration,
            operations_count=success_count,
            success_rate=success_rate,
            average_latency_ms=avg_latency,
            throughput_ops_per_second=throughput,
            memory_usage_mb=memory_delta,
            cost_efficiency_score=1.0,  # Alerting doesn't have direct cost
            detailed_metrics={
                'latency_p50': statistics.median(latencies) if latencies else 0,
                'latency_p95': self._percentile(latencies, 95) if latencies else 0,
                'latency_p99': self._percentile(latencies, 99) if latencies else 0,
                'total_alerts_generated': len(alert_manager.alert_history),
                'error_count': len(errors)
            },
            error_details=errors[:10]
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]


class DashboardGenerationBenchmark(PerformanceBenchmark):
    """Benchmark for dashboard generation performance."""
    
    def __init__(self):
        super().__init__("Dashboard Generation", BenchmarkType.LATENCY)
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run dashboard generation benchmark."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        dashboard = create_apify_dashboard(monitor, alert_manager)
        
        # Populate some test data
        monitor.start_monitoring()
        for i in range(100):
            session_id = f"dashboard_test_{i}"
            monitor.log_scraper_start(ScraperType.GOOGLE_PLACES, session_id, {})
            monitor.log_scraper_complete(session_id, {
                "status": "success",
                "results": [1, 2, 3],
                "estimated_cost": 25.0
            })
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        latencies = []
        errors = []
        
        try:
            for i in range(config.target_operations):
                gen_start = time.time()
                
                try:
                    # Test different dashboard formats
                    format_type = ["json", "html", "text"][i % 3]
                    
                    if format_type == "json":
                        dashboard.generate_dashboard_data()
                    elif format_type == "html":
                        dashboard.generate_html_dashboard()
                    else:
                        dashboard.generate_text_dashboard()
                    
                    gen_end = time.time()
                    latencies.append((gen_end - gen_start) * 1000)
                    
                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Error in dashboard generation {i}: {e}")
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            monitor.stop_monitoring()
        
        # Calculate metrics
        duration = end_time - start_time
        success_count = len(latencies)
        success_rate = success_count / config.target_operations
        avg_latency = statistics.mean(latencies) if latencies else 0
        throughput = success_count / duration if duration > 0 else 0
        memory_delta = end_memory - start_memory
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            duration_seconds=duration,
            operations_count=success_count,
            success_rate=success_rate,
            average_latency_ms=avg_latency,
            throughput_ops_per_second=throughput,
            memory_usage_mb=memory_delta,
            cost_efficiency_score=1.0,  # Dashboard generation doesn't have direct cost
            detailed_metrics={
                'latency_p50': statistics.median(latencies) if latencies else 0,
                'latency_p95': self._percentile(latencies, 95) if latencies else 0,
                'latency_p99': self._percentile(latencies, 99) if latencies else 0,
                'error_count': len(errors)
            },
            error_details=errors[:10]
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]


class ConcurrentLoadBenchmark(PerformanceBenchmark):
    """Benchmark for concurrent load handling."""
    
    def __init__(self):
        super().__init__("Concurrent Load", BenchmarkType.CONCURRENT_LOAD)
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run concurrent load benchmark."""
        monitor = create_apify_monitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        results = []
        errors = []
        
        def worker_task(worker_id: int, operations_per_worker: int):
            """Task for each worker thread."""
            worker_results = []
            worker_errors = []
            
            for i in range(operations_per_worker):
                session_id = f"concurrent_worker_{worker_id}_op_{i}"
                
                try:
                    op_start = time.time()
                    
                    monitor.log_scraper_start(ScraperType.GOOGLE_MAPS_CONTACTS, session_id, {
                        "worker_id": worker_id,
                        "operation": i
                    })
                    
                    # Simulate variable processing time
                    time.sleep(0.001 + (i % 5) * 0.001)
                    
                    monitor.log_scraper_complete(session_id, {
                        "status": "success",
                        "results": [f"result_{j}" for j in range(3)],
                        "estimated_cost": 15.0
                    })
                    
                    op_end = time.time()
                    worker_results.append((op_end - op_start) * 1000)
                    
                except Exception as e:
                    worker_errors.append(str(e))
            
            return worker_results, worker_errors
        
        try:
            # Calculate operations per worker
            operations_per_worker = config.target_operations // config.concurrent_workers
            
            # Run concurrent workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
                futures = [
                    executor.submit(worker_task, worker_id, operations_per_worker)
                    for worker_id in range(config.concurrent_workers)
                ]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures, timeout=config.timeout_seconds):
                    try:
                        worker_results, worker_errors = future.result()
                        results.extend(worker_results)
                        errors.extend(worker_errors)
                    except Exception as e:
                        errors.append(f"Worker error: {e}")
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            monitor.stop_monitoring()
        
        # Calculate metrics
        duration = end_time - start_time
        success_count = len(results)
        success_rate = success_count / config.target_operations
        avg_latency = statistics.mean(results) if results else 0
        throughput = success_count / duration if duration > 0 else 0
        memory_delta = end_memory - start_memory
        
        total_cost = success_count * 15.0
        cost_efficiency = success_count / total_cost if total_cost > 0 else 0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            duration_seconds=duration,
            operations_count=success_count,
            success_rate=success_rate,
            average_latency_ms=avg_latency,
            throughput_ops_per_second=throughput,
            memory_usage_mb=memory_delta,
            cost_efficiency_score=cost_efficiency,
            detailed_metrics={
                'concurrent_workers': config.concurrent_workers,
                'operations_per_worker': operations_per_worker,
                'latency_p50': statistics.median(results) if results else 0,
                'latency_p95': self._percentile(results, 95) if results else 0,
                'latency_p99': self._percentile(results, 99) if results else 0,
                'error_count': len(errors),
                'total_threads_used': config.concurrent_workers
            },
            error_details=errors[:10]
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]


class MemoryStressBenchmark(PerformanceBenchmark):
    """Benchmark for memory usage under stress."""
    
    def __init__(self):
        super().__init__("Memory Stress", BenchmarkType.STRESS_TEST)
    
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run memory stress benchmark."""
        monitor = create_apify_monitor()
        alert_manager = create_alert_manager()
        
        monitor.start_monitoring()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        memory_samples = []
        peak_memory = start_memory
        errors = []
        
        # Memory monitoring thread
        memory_monitoring_active = True
        
        def memory_monitor():
            while memory_monitoring_active:
                current_memory = self._get_memory_usage()
                memory_samples.append(current_memory)
                nonlocal peak_memory
                peak_memory = max(peak_memory, current_memory)
                time.sleep(config.memory_monitoring_interval)
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            # Generate high volume of events to stress memory
            for i in range(config.target_operations):
                session_id = f"memory_stress_{i}"
                
                try:
                    # Generate complex monitoring data
                    monitor.log_scraper_start(ScraperType.LINKEDIN_PREMIUM, session_id, {
                        "large_data": [f"data_item_{j}" for j in range(100)],  # Large data
                        "metadata": {"iteration": i, "timestamp": time.time()}
                    })
                    
                    # Generate large result set
                    large_results = {
                        "status": "success",
                        "results": [
                            {
                                "id": f"result_{j}",
                                "data": f"large_data_field_{j}" * 10,  # Large data per result
                                "metadata": {"index": j, "batch": i}
                            }
                            for j in range(50)  # 50 results per operation
                        ],
                        "estimated_cost": 30.0
                    }
                    
                    monitor.log_scraper_complete(session_id, large_results)
                    
                    # Trigger alert checks periodically
                    if i % 10 == 0:
                        monitoring_data = monitor.get_real_time_status()
                        alert_manager.check_alerts(monitoring_data)
                    
                    # Force garbage collection periodically
                    if i % 100 == 0:
                        gc.collect()
                
                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Error in memory stress test {i}: {e}")
        
        finally:
            memory_monitoring_active = False
            monitor_thread.join(timeout=1.0)
            end_time = time.time()
            end_memory = self._get_memory_usage()
            monitor.stop_monitoring()
        
        # Calculate metrics
        duration = end_time - start_time
        success_count = config.target_operations - len(errors)
        success_rate = success_count / config.target_operations
        throughput = success_count / duration if duration > 0 else 0
        memory_delta = end_memory - start_memory
        
        # Memory analysis
        avg_memory = statistics.mean(memory_samples) if memory_samples else start_memory
        memory_std = statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_type=self.benchmark_type,
            duration_seconds=duration,
            operations_count=success_count,
            success_rate=success_rate,
            average_latency_ms=0,  # Not applicable for memory test
            throughput_ops_per_second=throughput,
            memory_usage_mb=memory_delta,
            cost_efficiency_score=1.0,  # Not applicable
            detailed_metrics={
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'peak_memory_mb': peak_memory,
                'average_memory_mb': avg_memory,
                'memory_std_mb': memory_std,
                'memory_samples_count': len(memory_samples),
                'error_count': len(errors),
                'gc_collections': gc.get_count()
            },
            error_details=errors[:10]
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class PerformanceTestSuite:
    """Comprehensive performance test suite."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.benchmarks = [
            MonitoringThroughputBenchmark(),
            AlertingLatencyBenchmark(),
            DashboardGenerationBenchmark(),
            ConcurrentLoadBenchmark(),
            MemoryStressBenchmark()
        ]
        self.results: List[BenchmarkResult] = []
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info("Starting performance benchmark suite")
        
        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            
            try:
                # Force garbage collection before each benchmark
                gc.collect()
                
                result = benchmark.run(self.config)
                self.results.append(result)
                
                logger.info(f"Completed {benchmark.name}: "
                           f"{result.throughput_ops_per_second:.2f} ops/sec, "
                           f"{result.average_latency_ms:.2f}ms avg latency")
                
                # Cleanup
                benchmark.cleanup()
                
                # Small delay between benchmarks
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark.name}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    benchmark_type=benchmark.benchmark_type,
                    duration_seconds=0,
                    operations_count=0,
                    success_rate=0.0,
                    average_latency_ms=0.0,
                    throughput_ops_per_second=0.0,
                    memory_usage_mb=0.0,
                    cost_efficiency_score=0.0,
                    error_details=[str(e)]
                )
                self.results.append(error_result)
        
        logger.info("Performance benchmark suite completed")
        return self.results
    
    def run_specific_benchmark(self, benchmark_type: BenchmarkType) -> Optional[BenchmarkResult]:
        """Run a specific benchmark type."""
        for benchmark in self.benchmarks:
            if benchmark.benchmark_type == benchmark_type:
                logger.info(f"Running specific benchmark: {benchmark.name}")
                try:
                    result = benchmark.run(self.config)
                    self.results.append(result)
                    return result
                except Exception as e:
                    logger.error(f"Error running benchmark {benchmark.name}: {e}")
                    return None
        
        logger.warning(f"No benchmark found for type: {benchmark_type}")
        return None
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Overall statistics
        total_operations = sum(r.operations_count for r in self.results)
        average_throughput = statistics.mean([r.throughput_ops_per_second for r in self.results if r.throughput_ops_per_second > 0])
        average_latency = statistics.mean([r.average_latency_ms for r in self.results if r.average_latency_ms > 0])
        overall_success_rate = statistics.mean([r.success_rate for r in self.results])
        total_memory_usage = sum(r.memory_usage_mb for r in self.results)
        
        # Benchmark breakdown
        benchmark_details = []
        for result in self.results:
            benchmark_details.append({
                'name': result.benchmark_name,
                'type': result.benchmark_type.value,
                'duration_seconds': result.duration_seconds,
                'operations_count': result.operations_count,
                'success_rate': result.success_rate,
                'throughput_ops_per_second': result.throughput_ops_per_second,
                'average_latency_ms': result.average_latency_ms,
                'memory_usage_mb': result.memory_usage_mb,
                'cost_efficiency_score': result.cost_efficiency_score,
                'error_count': len(result.error_details),
                'detailed_metrics': result.detailed_metrics
            })
        
        # Performance assessment
        performance_grade = self._assess_performance()
        
        return {
            'timestamp': time.time(),
            'config': {
                'target_operations': self.config.target_operations,
                'concurrent_workers': self.config.concurrent_workers,
                'timeout_seconds': self.config.timeout_seconds
            },
            'summary': {
                'total_benchmarks': len(self.results),
                'total_operations': total_operations,
                'average_throughput_ops_per_second': average_throughput,
                'average_latency_ms': average_latency,
                'overall_success_rate': overall_success_rate,
                'total_memory_usage_mb': total_memory_usage,
                'performance_grade': performance_grade
            },
            'benchmark_results': benchmark_details,
            'recommendations': self._generate_recommendations()
        }
    
    def _assess_performance(self) -> str:
        """Assess overall performance and assign grade."""
        if not self.results:
            return "N/A"
        
        # Performance criteria
        score = 100
        
        # Deduct points for low success rates
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        if avg_success_rate < 0.95:
            score -= (0.95 - avg_success_rate) * 200  # Harsh penalty for failures
        
        # Deduct points for high latency
        latencies = [r.average_latency_ms for r in self.results if r.average_latency_ms > 0]
        if latencies:
            avg_latency = statistics.mean(latencies)
            if avg_latency > 100:  # 100ms threshold
                score -= min(50, (avg_latency - 100) / 10)  # Cap at 50 points
        
        # Deduct points for low throughput
        throughputs = [r.throughput_ops_per_second for r in self.results if r.throughput_ops_per_second > 0]
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput < 100:  # 100 ops/sec threshold
                score -= min(30, (100 - avg_throughput) / 5)  # Cap at 30 points
        
        # Grade assignment
        score = max(0, min(100, score))
        
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Failing)"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if not self.results:
            return ["No data available for recommendations"]
        
        # Check success rates
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        if avg_success_rate < 0.95:
            recommendations.append(f"Success rate ({avg_success_rate:.1%}) is below optimal. Check for error handling improvements.")
        
        # Check latencies
        latencies = [r.average_latency_ms for r in self.results if r.average_latency_ms > 0]
        if latencies:
            avg_latency = statistics.mean(latencies)
            if avg_latency > 100:
                recommendations.append(f"Average latency ({avg_latency:.1f}ms) is high. Consider performance optimizations.")
        
        # Check memory usage
        memory_usage = [r.memory_usage_mb for r in self.results if r.memory_usage_mb > 0]
        if memory_usage:
            total_memory = sum(memory_usage)
            if total_memory > 500:  # 500MB threshold
                recommendations.append(f"Total memory usage ({total_memory:.1f}MB) is high. Review memory management.")
        
        # Check concurrent performance
        concurrent_results = [r for r in self.results if r.benchmark_type == BenchmarkType.CONCURRENT_LOAD]
        if concurrent_results:
            concurrent_result = concurrent_results[0]
            if concurrent_result.throughput_ops_per_second < 50:
                recommendations.append("Concurrent load performance is low. Consider optimizing for multi-threading.")
        
        # Error analysis
        error_counts = [len(r.error_details) for r in self.results]
        if sum(error_counts) > 0:
            recommendations.append("Errors detected during benchmarks. Review error logs for improvement opportunities.")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges. No immediate improvements needed.")
        
        return recommendations
    
    def export_results(self, file_path: Path):
        """Export benchmark results to file."""
        report = self.generate_performance_report()
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            # Export as text
            with open(file_path, 'w') as f:
                f.write("APIFY AUTOMATION PERFORMANCE BENCHMARK REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Summary
                summary = report['summary']
                f.write("SUMMARY:\n")
                f.write(f"  Total Benchmarks: {summary['total_benchmarks']}\n")
                f.write(f"  Total Operations: {summary['total_operations']}\n")
                f.write(f"  Average Throughput: {summary['average_throughput_ops_per_second']:.2f} ops/sec\n")
                f.write(f"  Average Latency: {summary['average_latency_ms']:.2f} ms\n")
                f.write(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}\n")
                f.write(f"  Performance Grade: {summary['performance_grade']}\n\n")
                
                # Individual benchmarks
                f.write("BENCHMARK DETAILS:\n")
                for benchmark in report['benchmark_results']:
                    f.write(f"  {benchmark['name']}:\n")
                    f.write(f"    Throughput: {benchmark['throughput_ops_per_second']:.2f} ops/sec\n")
                    f.write(f"    Latency: {benchmark['average_latency_ms']:.2f} ms\n")
                    f.write(f"    Success Rate: {benchmark['success_rate']:.1%}\n")
                    f.write(f"    Memory Usage: {benchmark['memory_usage_mb']:.2f} MB\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                for rec in report['recommendations']:
                    f.write(f"  - {rec}\n")
        
        logger.info(f"Benchmark results exported to {file_path}")


def run_performance_tests(
    target_operations: int = 1000,
    concurrent_workers: int = 5,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run complete performance test suite."""
    config = BenchmarkConfig(
        target_operations=target_operations,
        concurrent_workers=concurrent_workers,
        output_directory=output_dir
    )
    
    test_suite = PerformanceTestSuite(config)
    test_suite.run_all_benchmarks()
    
    report = test_suite.generate_performance_report()
    
    if output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"performance_report_{timestamp}.json"
        test_suite.export_results(report_file)
    
    return report


if __name__ == "__main__":
    # Run basic performance tests
    import sys
    from pathlib import Path
    
    output_dir = Path("./benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Running Apify automation performance benchmarks...")
    report = run_performance_tests(
        target_operations=500,  # Reduced for demo
        concurrent_workers=3,
        output_dir=output_dir
    )
    
    print(f"\nPerformance Grade: {report['summary']['performance_grade']}")
    print(f"Average Throughput: {report['summary']['average_throughput_ops_per_second']:.2f} ops/sec")
    print(f"Overall Success Rate: {report['summary']['overall_success_rate']:.1%}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")