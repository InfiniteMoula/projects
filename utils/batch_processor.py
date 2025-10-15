#!/usr/bin/env python3
"""
Batch processing optimization for memory-efficient and parallel data processing.

This module provides optimized batch processing capabilities for large datasets,
including memory management, concurrent execution, and progress tracking.
"""

import asyncio
import gc
import logging
import time
from typing import AsyncGenerator, Callable, Any, List, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import threading

from .adaptive_controller import AdaptiveController, AdaptiveState

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_batches: int = 0
    processed_batches: int = 0
    failed_batches: int = 0
    total_items: int = 0
    processed_items: int = 0
    start_time: float = 0
    end_time: float = 0
    memory_peak_mb: float = 0
    avg_batch_time: float = 0


class MemoryMonitor:
    """Monitor memory usage during batch processing."""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0
        self.monitoring = False
        self._thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Monitor memory usage continuously."""
        while self.monitoring:
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                
                if memory_mb > self.max_memory_mb:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB (limit: {self.max_memory_mb}MB)")
                    gc.collect()  # Force garbage collection
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
            time.sleep(1)  # Check every second


class BatchProcessor:
    """
    Optimized batch processor for memory-efficient parallel processing.
    """
    
    def __init__(
        self,
        batch_size: int = 50,
        max_concurrent: int = 3,
        max_memory_mb: int = 4096,
        rate_limit_delay: float = 1.0,
        retry_failed: bool = True,
        max_retries: int = 2,
        adaptive_controller: Optional[AdaptiveController] = None,
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.max_memory_mb = max_memory_mb
        self.rate_limit_delay = rate_limit_delay
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.adaptive_controller = adaptive_controller

        self.current_batch_size = adaptive_controller.current_chunk_size if adaptive_controller else batch_size
        self.current_max_concurrent = adaptive_controller.current_concurrency if adaptive_controller else max_concurrent

        self.stats = BatchStats()
        self.memory_monitor = MemoryMonitor(max_memory_mb)

    def _sync_adaptive_defaults(self) -> None:
        if self.adaptive_controller:
            self.current_batch_size = self.adaptive_controller.current_chunk_size
            self.current_max_concurrent = self.adaptive_controller.current_concurrency
        else:
            self.current_batch_size = self.batch_size
            self.current_max_concurrent = self.max_concurrent

    async def process_batches_async(
        self,
        df: pd.DataFrame,
        processor_func: Callable,
        *args,
        **kwargs
    ) -> AsyncGenerator[List[Any], None]:
        """
        Process DataFrame in batches asynchronously with memory optimization.
        
        Args:
            df: Input DataFrame to process
            processor_func: Async function to process each batch
            *args, **kwargs: Additional arguments for processor_func
            
        Yields:
            Results from each processed batch
        """
        logger.info(f"Starting batch processing: {len(df)} items in batches of {self.batch_size}")

        self._sync_adaptive_defaults()

        # Initialize stats
        self.stats = BatchStats()
        self.stats.total_items = len(df)
        self.stats.total_batches = 0
        self.stats.start_time = time.time()

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        try:
            semaphore = asyncio.Semaphore(max(1, self.current_max_concurrent))
            tasks: List[asyncio.Task] = []
            batch_index = 0
            start_idx = 0
            chunk_started = time.time()

            while start_idx < len(df):
                if not tasks:
                    chunk_started = time.time()

                end_idx = min(start_idx + self.current_batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx].copy()
                self.stats.total_batches += 1

                task = self._process_single_batch_async(
                    batch_df,
                    batch_index,
                    self.stats.total_batches,
                    processor_func,
                    semaphore,
                    *args,
                    **kwargs,
                )
                tasks.append(task)
                batch_index += 1
                start_idx = end_idx

                if len(tasks) >= self.current_max_concurrent or start_idx >= len(df):
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    failures = 0
                    processed_items = 0
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Batch processing error: {result}")
                            self.stats.failed_batches += 1
                            failures += 1
                        else:
                            self.stats.processed_batches += 1
                            self.stats.processed_items += len(result)
                            processed_items += len(result)
                            yield result

                    tasks.clear()
                    gc.collect()

                    duration = max(time.time() - chunk_started, 1e-6)
                    state = self._run_adaptive_cycle(
                        failures=failures,
                        total=len(results),
                        processed_items=processed_items,
                        duration=duration,
                    )
                    if state and state.concurrency_changed:
                        semaphore = asyncio.Semaphore(max(1, self.current_max_concurrent))

                    if start_idx < len(df):
                        await asyncio.sleep(self.rate_limit_delay)

        finally:
            # Cleanup
            self.memory_monitor.stop_monitoring()
            self.stats.end_time = time.time()
            self.stats.memory_peak_mb = self.memory_monitor.peak_memory_mb
            
            total_time = self.stats.end_time - self.stats.start_time
            self.stats.avg_batch_time = total_time / max(self.stats.processed_batches, 1)
            
            logger.info(f"Batch processing completed: {self.stats.processed_batches}/{self.stats.total_batches} batches")
            logger.info(f"Peak memory usage: {self.stats.memory_peak_mb:.1f}MB")
            logger.info(f"Average batch time: {self.stats.avg_batch_time:.2f}s")
    
    async def _process_single_batch_async(
        self,
        batch_df: pd.DataFrame,
        batch_num: int,
        total_batches: int,
        processor_func: Callable,
        semaphore: asyncio.Semaphore,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a single batch with concurrency control and retry logic."""
        
        async with semaphore:
            retries = 0
            last_exception = None
            
            while retries <= self.max_retries:
                try:
                    start_time = time.time()
                    logger.debug(f"Processing batch {batch_num + 1}/{total_batches} (size: {len(batch_df)})")

                    # Call the processor function
                    result = await processor_func(batch_df, *args, **kwargs)

                    processing_time = time.time() - start_time
                    logger.debug(f"Batch {batch_num + 1}/{total_batches} completed in {processing_time:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    retries += 1
                    
                    if retries <= self.max_retries and self.retry_failed:
                        logger.warning(f"Batch {batch_num + 1} failed (attempt {retries}): {e}")
                        await asyncio.sleep(retries * 2)  # Exponential backoff
                    else:
                        logger.error(f"Batch {batch_num + 1} failed after {retries} attempts: {e}")
                        break
            
            # If all retries failed, return empty result
            if last_exception:
                raise last_exception
            
            return []
    
    def process_batches_sync(
        self,
        df: pd.DataFrame,
        processor_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process DataFrame in batches synchronously with threading.
        
        Args:
            df: Input DataFrame to process
            processor_func: Function to process each batch
            *args, **kwargs: Additional arguments for processor_func
            
        Returns:
            Combined results from all batches
        """
        logger.info(f"Starting sync batch processing: {len(df)} items in batches of {self.batch_size}")

        self._sync_adaptive_defaults()

        # Initialize stats
        self.stats = BatchStats()
        self.stats.total_items = len(df)
        self.stats.total_batches = 0
        self.stats.start_time = time.time()

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        all_results = []

        try:
            start_idx = 0
            batch_index = 0

            while start_idx < len(df):
                chunk_start = time.time()
                scheduled_batches: List[Tuple[pd.DataFrame, int, int]] = []

                while (
                    len(scheduled_batches) < max(1, self.current_max_concurrent)
                    and start_idx < len(df)
                ):
                    end_idx = min(start_idx + self.current_batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx].copy()
                    self.stats.total_batches += 1
                    scheduled_batches.append((batch_df, batch_index, self.stats.total_batches))
                    batch_index += 1
                    start_idx = end_idx

                failures = 0
                processed_items = 0

                with ThreadPoolExecutor(
                    max_workers=max(1, min(self.current_max_concurrent, len(scheduled_batches)))
                ) as executor:
                    future_map = {
                        executor.submit(
                            self._process_single_batch_sync,
                            batch_df,
                            batch_num,
                            total_hint,
                            processor_func,
                            *args,
                            **kwargs,
                        ): batch_num
                        for batch_df, batch_num, total_hint in scheduled_batches
                    }

                    for future in as_completed(future_map):
                        batch_num = future_map[future]
                        try:
                            result = future.result()
                            self.stats.processed_batches += 1
                            if result:
                                self.stats.processed_items += len(result)
                                processed_items += len(result)
                                all_results.extend(result)

                            logger.debug(f"Batch {batch_num + 1} completed")

                        except Exception as e:
                            logger.error(f"Batch {batch_num + 1} failed: {e}")
                            self.stats.failed_batches += 1
                            failures += 1

                gc.collect()
                state = self._run_adaptive_cycle(
                    failures=failures,
                    total=len(scheduled_batches),
                    processed_items=processed_items,
                    duration=max(time.time() - chunk_start, 1e-6),
                )
                if state and state.concurrency_changed:
                    # Ensure future iterations honour the new concurrency
                    self.current_max_concurrent = state.concurrency

                if start_idx < len(df):
                    time.sleep(self.rate_limit_delay)

        finally:
            # Cleanup
            self.memory_monitor.stop_monitoring()
            self.stats.end_time = time.time()
            self.stats.memory_peak_mb = self.memory_monitor.peak_memory_mb
            
            total_time = self.stats.end_time - self.stats.start_time
            self.stats.avg_batch_time = total_time / max(self.stats.processed_batches, 1)
            
            logger.info(f"Sync batch processing completed: {self.stats.processed_batches}/{self.stats.total_batches} batches")
            logger.info(f"Peak memory usage: {self.stats.memory_peak_mb:.1f}MB")
            logger.info(f"Average batch time: {self.stats.avg_batch_time:.2f}s")
        
        return all_results
    
    def _process_single_batch_sync(
        self,
        batch_df: pd.DataFrame,
        batch_num: int,
        total_batches: int,
        processor_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a single batch synchronously with retry logic."""
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                start_time = time.time()
                logger.debug(f"Processing batch {batch_num + 1}/{total_batches} (size: {len(batch_df)})")

                # Call the processor function
                result = processor_func(batch_df, *args, **kwargs)

                processing_time = time.time() - start_time
                logger.debug(f"Batch {batch_num + 1}/{total_batches} completed in {processing_time:.2f}s")
                
                return result
                
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries <= self.max_retries and self.retry_failed:
                    logger.warning(f"Batch {batch_num + 1} failed (attempt {retries}): {e}")
                    time.sleep(retries * 2)  # Exponential backoff
                else:
                    logger.error(f"Batch {batch_num + 1} failed after {retries} attempts: {e}")
                    break
        
        # If all retries failed, return empty result
        if last_exception:
            raise last_exception

        return []

    def _run_adaptive_cycle(
        self,
        *,
        failures: int,
        total: int,
        processed_items: int,
        duration: float,
    ) -> Optional[AdaptiveState]:
        if not self.adaptive_controller or total <= 0:
            return None

        error_rate = failures / total if total > 0 else 0.0
        req_per_min = None
        if processed_items > 0 and duration > 0:
            req_per_min = (processed_items / duration) * 60.0

        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        state = self.adaptive_controller.observe(
            error_rate=error_rate,
            req_per_min=req_per_min,
            ram_used=ram_gb,
        )
        self._apply_adaptive_state(state)
        return state

    def _apply_adaptive_state(self, state: Optional[AdaptiveState]) -> None:
        if not state:
            return

        if state.concurrency != self.current_max_concurrent:
            self.current_max_concurrent = state.concurrency
            if state.concurrency_changed:
                logger.info("Adaptive controller set max_concurrent=%s", state.concurrency)

        if state.chunk_size != self.current_batch_size:
            self.current_batch_size = state.chunk_size
            if state.chunk_size_changed:
                logger.info("Adaptive controller set batch_size=%s", state.chunk_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_batches": self.stats.total_batches,
            "processed_batches": self.stats.processed_batches,
            "failed_batches": self.stats.failed_batches,
            "total_items": self.stats.total_items,
            "processed_items": self.stats.processed_items,
            "success_rate": self.stats.processed_batches / max(self.stats.total_batches, 1),
            "processing_time": self.stats.end_time - self.stats.start_time if self.stats.end_time else 0,
            "memory_peak_mb": self.stats.memory_peak_mb,
            "avg_batch_time": self.stats.avg_batch_time,
            "throughput_items_per_sec": self.stats.processed_items / max(self.stats.end_time - self.stats.start_time, 1) if self.stats.end_time else 0
        }


def create_batch_processor(
    batch_size: int = 50,
    max_concurrent: int = 3,
    config: Optional[Dict] = None,
    adaptive_controller: Optional[AdaptiveController] = None,
) -> BatchProcessor:
    """
    Factory function to create a configured BatchProcessor.
    
    Args:
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent batch processing
        config: Optional configuration dictionary
        
    Returns:
        Configured BatchProcessor instance
    """
    if config:
        batch_size = config.get("batch_size", batch_size)
        max_concurrent = config.get("max_concurrent", max_concurrent)
        max_memory_mb = config.get("max_memory_mb", 4096)
        rate_limit_delay = config.get("rate_limit_delay", 1.0)
        retry_failed = config.get("retry_failed", True)
        max_retries = config.get("max_retries", 2)
    else:
        max_memory_mb = 4096
        rate_limit_delay = 1.0
        retry_failed = True
        max_retries = 2
    
    return BatchProcessor(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        max_memory_mb=max_memory_mb,
        rate_limit_delay=rate_limit_delay,
        retry_failed=retry_failed,
        max_retries=max_retries,
        adaptive_controller=adaptive_controller,
    )


# Utility functions for streaming processing
async def stream_process_file(
    file_path: str,
    chunk_size: int = 1000,
    processor_func: Callable = None
) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Stream process a large file in chunks.
    
    Args:
        file_path: Path to the file to process
        chunk_size: Number of rows per chunk
        processor_func: Optional function to process each chunk
        
    Yields:
        Processed DataFrame chunks
    """
    logger.info(f"Starting stream processing of {file_path} with chunk size {chunk_size}")
    
    try:
        if file_path.endswith('.parquet'):
            # Use pandas chunking for parquet files
            for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
                if processor_func:
                    chunk = await processor_func(chunk) if asyncio.iscoroutinefunction(processor_func) else processor_func(chunk)
                yield chunk
                
        elif file_path.endswith('.csv'):
            # Use pandas chunking for CSV files
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                if processor_func:
                    chunk = await processor_func(chunk) if asyncio.iscoroutinefunction(processor_func) else processor_func(chunk)
                yield chunk
                
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return
            
    except Exception as e:
        logger.error(f"Error streaming file {file_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def example_processor(batch_df: pd.DataFrame) -> List[Dict]:
        """Example batch processor function."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Return processed results
        return [{"processed": True, "count": len(batch_df)}]
    
    async def main():
        # Create sample data
        df = pd.DataFrame({
            "id": range(100),
            "value": [f"item_{i}" for i in range(100)]
        })
        
        # Create batch processor
        processor = create_batch_processor(batch_size=10, max_concurrent=3)
        
        # Process batches
        results = []
        async for batch_results in processor.process_batches_async(df, example_processor):
            results.extend(batch_results)
        
        print(f"Processed {len(results)} items")
        print(f"Stats: {processor.get_stats()}")
    
    asyncio.run(main())