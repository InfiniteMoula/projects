#!/usr/bin/env python3
"""
Demonstration script for NAF Reference Collection module

This script shows how to use the collect.nafreference module to download
and process NAF (Nomenclature d'Activités Françaises) reference data.
"""

import sys
import tempfile
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collect.nafreference import run

def demo_naf_collection():
    """Demonstrate NAF reference data collection."""
    
    # Enable logging to see what's happening
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    print("=== NAF Reference Collection Demo ===\n")
    
    # Demo 1: Dry run
    print("1. Dry run (no actual download):")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"nafreference": {"naf_code": "01"}}
        ctx = {"outdir": tmpdir, "dry_run": True}
        result = run(cfg, ctx)
        print(f"   Result: {result}")
        print()
    
    # Demo 2: Download agriculture sector (NAF 01)
    print("2. Download agriculture sector data (NAF code 01):")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "nafreference": {
                "naf_code": "01",
                "source": "fallback",
                "timeout": 30
            }
        }
        ctx = {"outdir": tmpdir}
        result = run(cfg, ctx)
        print(f"   Result: {result}")
        
        if result["status"] == "OK":
            import pandas as pd
            df = pd.read_parquet(result["file"])
            print(f"   Downloaded {len(df)} records")
            print("   Sample data:")
            print(df.head(3).to_string(index=False))
        print()
    
    # Demo 3: Download all NAF data (no filter)
    print("3. Download all NAF reference data:")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "nafreference": {
                "source": "fallback",
                "timeout": 60
            }
        }
        ctx = {"outdir": tmpdir}
        result = run(cfg, ctx)
        print(f"   Result: {result}")
        print()
    
    print("=== Demo completed ===")

if __name__ == "__main__":
    demo_naf_collection()