#!/usr/bin/env python3
"""
Test script for Phase 3 implementation.

This script demonstrates the newly implemented features:
- Enhanced progress tracking
- Batch processing capabilities
- Async scraper structure (basic functionality)
"""

import sys
import json
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.progress_tracker import ProgressTracker, create_progress_tracker
from utils.batch_processor import BatchProcessor, create_batch_processor


async def demo_async_processing():
    """Demonstrate async batch processing."""
    print("\n=== Phase 3 Demo: Async Batch Processing ===")
    
    # Create a simple data structure to simulate DataFrame
    class SimpleDataFrame:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def iloc(self, start, end):
            return SimpleDataFrame(self.data[start:end])
        
        def copy(self):
            return SimpleDataFrame(self.data.copy())
    
    # Sample data
    sample_data = [{"id": i, "value": f"item_{i}"} for i in range(50)]
    df = SimpleDataFrame(sample_data)
    
    # Create progress tracker
    tracker = create_progress_tracker("/tmp/phase3_demo")
    
    # Define phases for the demo
    tracker.define_phases([
        {"name": "initialization", "description": "Setup and preparation", "total_items": 1},
        {"name": "batch_processing", "description": "Processing data in batches", "total_items": len(df)},
        {"name": "finalization", "description": "Cleanup and results", "total_items": 1}
    ])
    
    # Progress callback
    def progress_callback(metrics):
        print(f"Progress: {metrics.completed_items}/{metrics.total_items} "
              f"({metrics.completed_items/max(metrics.total_items, 1)*100:.1f}%) "
              f"- Phase: {metrics.current_phase} - Speed: {metrics.items_per_second:.1f} items/sec")
    
    tracker.add_callback(progress_callback)
    
    # Start initialization phase
    tracker.start_phase("initialization", 1)
    await asyncio.sleep(0.5)  # Simulate initialization
    tracker.update_progress(1, current_item="system_setup")
    tracker.complete_phase("initialization")
    
    # Batch processing phase
    tracker.start_phase("batch_processing", len(df))
    
    # Create batch processor
    processor = create_batch_processor(batch_size=10, max_concurrent=2)
    
    # Async batch processor function
    async def process_batch(batch_df):
        """Process a single batch."""
        await asyncio.sleep(0.2)  # Simulate processing time
        results = []
        for item in batch_df.data:
            results.append({
                "processed_id": item["id"],
                "processed_value": f"processed_{item['value']}",
                "status": "success"
            })
        return results
    
    # Process batches
    all_results = []
    async for batch_results in processor.process_batches_async(df, process_batch):
        all_results.extend(batch_results)
        tracker.update_progress(len(batch_results))
    
    tracker.complete_phase("batch_processing")
    
    # Finalization
    tracker.start_phase("finalization", 1)
    await asyncio.sleep(0.2)
    tracker.update_progress(1, current_item="saving_results")
    tracker.complete_phase("finalization")
    
    # Complete processing
    tracker.complete_processing()
    
    # Print results
    print(f"\n✓ Processed {len(all_results)} items successfully")
    print(f"✓ Batch processing stats: {processor.get_stats()}")
    
    # Print final summary
    summary = tracker.get_summary()
    print(f"\n=== Processing Summary ===")
    print(f"Total time: {summary.get('total_time', 0):.2f} seconds")
    print(f"Success rate: {(1 - summary['overall']['error_rate']) * 100:.1f}%")
    print(f"Average throughput: {summary['overall']['items_per_second']:.1f} items/sec")
    
    return all_results


def demo_progress_tracking():
    """Demonstrate enhanced progress tracking."""
    print("\n=== Phase 3 Demo: Enhanced Progress Tracking ===")
    
    # Create progress tracker
    tracker = create_progress_tracker("/tmp/progress_demo")
    
    # Define multi-phase processing
    phases = [
        {"name": "data_loading", "description": "Loading input data", "total_items": 100},
        {"name": "validation", "description": "Validating data quality", "total_items": 100},
        {"name": "enrichment", "description": "Enriching with external data", "total_items": 100}
    ]
    
    tracker.define_phases(phases)
    
    # Progress callback for real-time updates
    def print_progress(metrics):
        phase = metrics.current_phase
        completed = metrics.completed_items
        total = metrics.total_items
        percentage = (completed / max(total, 1)) * 100
        eta = metrics.estimated_time_remaining
        
        print(f"[{phase.upper()}] {completed}/{total} ({percentage:.1f}%) "
              f"ETA: {eta:.0f}s Speed: {metrics.items_per_second:.1f}/s")
    
    tracker.add_callback(print_progress)
    
    # Simulate multi-phase processing
    for phase_info in phases:
        phase_name = phase_info["name"]
        total_items = phase_info["total_items"]
        
        tracker.start_phase(phase_name, total_items)
        
        # Simulate processing items
        for i in range(total_items):
            time.sleep(0.02)  # Simulate work
            
            # Occasionally simulate errors
            if i % 25 == 0 and i > 0:
                tracker.report_error(f"Simulated error at item {i}", f"item_{i}")
            else:
                tracker.update_progress(1, current_item=f"item_{i}")
        
        tracker.complete_phase(phase_name)
    
    tracker.complete_processing()
    
    # Print detailed summary
    summary = tracker.get_summary()
    print(f"\n=== Detailed Progress Summary ===")
    print(f"Overall completion: {summary['completion_percentage']:.1f}%")
    print(f"Total processing time: {summary.get('elapsed_time', 0):.2f}s")
    print(f"Error rate: {summary['overall']['error_rate']*100:.1f}%")
    
    # Phase-specific summaries
    print("\n--- Phase Details ---")
    for phase_name in [p["name"] for p in phases]:
        phase_summary = tracker.get_phase_summary(phase_name)
        if phase_summary:
            print(f"{phase_name}: {phase_summary['completion_percentage']:.1f}% complete "
                  f"in {phase_summary['elapsed_time']:.1f}s")


def demo_memory_efficient_processing():
    """Demonstrate memory-efficient batch processing."""
    print("\n=== Phase 3 Demo: Memory-Efficient Processing ===")
    
    # Create batch processor with memory monitoring
    processor = BatchProcessor(
        batch_size=20,
        max_concurrent=3,
        max_memory_mb=512,  # Low limit for demo
        rate_limit_delay=0.1
    )
    
    # Simulate large dataset
    large_dataset = []
    for i in range(200):
        # Simulate data with some complexity
        item = {
            "id": i,
            "data": f"large_data_item_{i}_" + "x" * 100,  # Simulate larger data
            "metadata": {"created": time.time(), "size": i * 10}
        }
        large_dataset.append(item)
    
    class LargeDataFrame:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def iloc(self, start, end):
            if hasattr(start, 'stop'):  # slice object
                return LargeDataFrame(self.data[start])
            else:
                return LargeDataFrame(self.data[start:end])
    
    df = LargeDataFrame(large_dataset)
    
    def process_batch_sync(batch_df):
        """Synchronous batch processor for demo."""
        time.sleep(0.1)  # Simulate processing
        results = []
        for item in batch_df.data:
            results.append({
                "processed_id": item["id"],
                "result": f"processed_{item['id']}",
                "memory_efficient": True
            })
        return results
    
    print(f"Processing {len(df)} items with memory monitoring...")
    
    # Process with memory monitoring
    start_time = time.time()
    results = processor.process_batches_sync(df, process_batch_sync)
    end_time = time.time()
    
    # Get processing statistics
    stats = processor.get_stats()
    
    print(f"\n✓ Memory-efficient processing completed!")
    print(f"✓ Processed {len(results)} items in {end_time - start_time:.2f}s")
    print(f"✓ Peak memory usage: {stats['memory_peak_mb']:.1f}MB")
    print(f"✓ Success rate: {stats['success_rate']*100:.1f}%")
    print(f"✓ Average batch time: {stats['avg_batch_time']:.2f}s")
    print(f"✓ Throughput: {stats['throughput_items_per_sec']:.1f} items/sec")


async def main():
    """Run all Phase 3 demos."""
    print("🚀 Phase 3 Implementation Demo")
    print("=" * 50)
    
    try:
        # Demo 1: Enhanced Progress Tracking
        demo_progress_tracking()
        
        # Demo 2: Memory-Efficient Processing
        demo_memory_efficient_processing()
        
        # Demo 3: Async Processing (if asyncio available)
        await demo_async_processing()
        
        print("\n" + "=" * 50)
        print("✅ Phase 3 Demo completed successfully!")
        print("\nImplemented features:")
        print("  ✓ Enhanced progress tracking with real-time metrics")
        print("  ✓ Memory-efficient batch processing")
        print("  ✓ Async processing framework")
        print("  ✓ Multi-phase processing support")
        print("  ✓ Performance monitoring and optimization")
        
        # Note about ML components
        print("\n📝 Note: ML components (extraction models, address classifier, ML optimizer)")
        print("   are implemented but require additional dependencies (pandas, sklearn, numpy)")
        print("   that are not available in this environment. They can be tested separately")
        print("   after installing the updated requirements.txt")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())