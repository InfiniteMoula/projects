#!/usr/bin/env python3
"""
Async Apify API agents for parallel business data enrichment.

This module provides async integration with Apify scrapers for parallel execution:
1. Google Places Crawler (compass/crawler-google-places)
2. Google Maps with Contact Details (lukaskrivka/google-maps-with-contact-details)
3. LinkedIn Premium Actor (bebity/linkedin-premium-actor)

Enables concurrent processing for improved performance and throughput.
"""

import asyncio
import json
import time
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import pandas as pd
import aiohttp
from apify_client import ApifyClient

from utils import io
from utils.parquet import ParquetBatchWriter
from utils.address_processor import AddressProcessor
from utils.company_name_processor import CompanyNameProcessor
from utils.quality_controller import GoogleMapsQualityController, LinkedInQualityController
from utils.retry_manager import LinkedInRetryManager, GoogleMapsRetryManager, RetryConfig
from utils.cost_manager import CostManager, ScraperType, create_cost_manager
from utils.dynamic_config import DynamicConfigurationManager, ConfigPriority, create_dynamic_config_manager
from utils.industry_optimizer import IndustryOptimizer, IndustryCategory, OptimizationStrategy, create_industry_optimizer
from utils.progress_tracker import ProgressTracker
from utils.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class AsyncApifyClient:
    """Async wrapper for Apify client operations."""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.apify.com/v2"
        
    async def call_actor_async(self, actor_id: str, run_input: dict, timeout: int = 600) -> dict:
        """Call Apify actor asynchronously."""
        async with aiohttp.ClientSession() as session:
            # Start actor run
            start_url = f"{self.base_url}/acts/{actor_id}/runs"
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            
            async with session.post(start_url, json=run_input, headers=headers) as response:
                if response.status != 201:
                    raise Exception(f"Failed to start actor: {response.status}")
                run_data = await response.json()
                run_id = run_data["data"]["id"]
            
            # Wait for completion
            status_url = f"{self.base_url}/actor-runs/{run_id}"
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                async with session.get(status_url, headers=headers) as response:
                    status_data = await response.json()
                    status = status_data["data"]["status"]
                    
                    if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                        break
                        
                await asyncio.sleep(5)  # Check every 5 seconds
            
            # Get results
            dataset_id = status_data["data"]["defaultDatasetId"]
            results_url = f"{self.base_url}/datasets/{dataset_id}/items"
            
            results = []
            async with session.get(results_url, headers=headers) as response:
                if response.status == 200:
                    results = await response.json()
                    
            return {
                "status": status,
                "results": results,
                "stats": status_data["data"]["stats"]
            }


async def run_async(cfg: dict, ctx: dict) -> dict:
    """
    Async version of the main Apify enrichment process.
    
    Runs scrapers in parallel for improved performance.
    """
    logger.info("Starting async Apify enrichment process...")
    
    # Initialize components
    progress_tracker = ProgressTracker(ctx.get("output_dir", "."))
    batch_processor = BatchProcessor(
        batch_size=cfg.get("batch_size", 10),
        max_concurrent=cfg.get("max_concurrent", 3)
    )
    
    # Load and validate input data
    try:
        df = _load_and_validate_input_async(cfg, ctx)
        if df is None or df.empty:
            return {"status": "error", "message": "No valid input data"}
            
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return {"status": "error", "message": str(e)}
    
    # Initialize Apify client
    apify_token = os.getenv("APIFY_TOKEN")
    if not apify_token:
        logger.error("APIFY_TOKEN environment variable not set")
        return {"status": "error", "message": "APIFY_TOKEN not configured"}
    
    async_client = AsyncApifyClient(apify_token)
    
    # Process in batches
    all_results = []
    total_rows = len(df)
    
    progress_tracker.start_processing(total_rows)
    
    async for batch_results in batch_processor.process_batches_async(
        df, _process_batch_async, async_client, cfg, ctx, progress_tracker
    ):
        all_results.extend(batch_results)
        progress_tracker.update_progress(len(batch_results))
    
    progress_tracker.complete_processing()
    
    # Combine and save results
    final_results = _combine_results_async(all_results, df)
    
    # Save output
    output_path = _save_results_async(final_results, cfg, ctx)
    
    logger.info(f"Async Apify enrichment completed. Results saved to: {output_path}")
    
    return {
        "status": "success",
        "output_path": output_path,
        "total_processed": len(all_results),
        "performance_stats": progress_tracker.get_stats()
    }


async def _process_batch_async(
    batch_df: pd.DataFrame, 
    client: AsyncApifyClient, 
    cfg: dict, 
    ctx: dict,
    progress_tracker: ProgressTracker
) -> List[Dict]:
    """Process a single batch of data asynchronously."""
    
    # Extract data for scrapers
    addresses = _extract_addresses(batch_df)
    companies = _extract_companies(batch_df)
    
    # Create async tasks for parallel execution
    tasks = []
    
    apify_config = cfg.get("apify", {})
    
    # Google Places task
    if apify_config.get("google_places", {}).get("enabled", True) and addresses:
        tasks.append(_run_google_places_async(addresses, client, apify_config.get("google_places", {})))
    
    # LinkedIn task (can run in parallel with Google Places)
    if apify_config.get("linkedin_premium", {}).get("enabled", True) and companies:
        tasks.append(_run_linkedin_premium_async(companies, client, apify_config.get("linkedin_premium", {})))
    
    # Execute tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    places_results = []
    linkedin_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed: {result}")
            continue
            
        if i == 0 and addresses:  # Google Places
            places_results = result
        elif (i == 1 and companies) or (i == 0 and not addresses):  # LinkedIn
            linkedin_results = result
    
    # Google Maps Contacts (depends on places results)
    contact_results = []
    if apify_config.get("google_maps_contacts", {}).get("enabled", True) and places_results:
        contact_results = await _run_google_maps_contacts_async(
            places_results, client, apify_config.get("google_maps_contacts", {})
        )
    
    # Combine batch results
    batch_results = []
    for _, row in batch_df.iterrows():
        result = {
            "original_data": row.to_dict(),
            "places": _match_places_to_row(row, places_results),
            "contacts": _match_contacts_to_row(row, contact_results),
            "linkedin": _match_linkedin_to_row(row, linkedin_results)
        }
        batch_results.append(result)
    
    return batch_results


async def _run_google_places_async(addresses: List[str], client: AsyncApifyClient, cfg: dict) -> List[Dict]:
    """Async version of Google Places Crawler."""
    actor_id = "compass/crawler-google-places"
    
    run_input = {
        "searchTerms": addresses,
        "language": "fr",
        "maxCrawledPlaces": cfg.get("max_places_per_search", 10),
        "exportPlaceUrls": True,
        "includeImages": False,
        "includeReviews": False,
        "maxReviews": 0,
    }
    
    logger.info(f"Starting async Google Places Crawler for {len(addresses)} addresses...")
    
    try:
        result = await client.call_actor_async(actor_id, run_input, timeout=cfg.get("timeout_seconds", 600))
        
        if result["status"] == "SUCCEEDED":
            logger.info(f"Google Places Crawler completed: {len(result['results'])} places found")
            return result["results"]
        else:
            logger.error(f"Google Places Crawler failed with status: {result['status']}")
            return []
            
    except Exception as e:
        logger.error(f"Google Places Crawler async execution failed: {e}")
        return []


async def _run_linkedin_premium_async(companies: List[str], client: AsyncApifyClient, cfg: dict) -> List[Dict]:
    """Async version of LinkedIn Premium Actor."""
    actor_id = "bebity/linkedin-premium-actor"
    
    search_terms = companies[:cfg.get("max_linkedin_searches", 20)]
    
    if not search_terms:
        return []
    
    run_input = {
        "searchTerms": search_terms,
        "language": "fr",
        "maxProfiles": cfg.get("max_profiles_per_company", 5),
        "filters": {
            "positions": cfg.get("filters", {}).get("positions", 
                ["CEO", "CFO", "Director", "Founder", "Gérant", "Directeur", "Président"])
        },
        "timeoutMs": cfg.get("timeout_seconds", 600) * 1000
    }
    
    logger.info(f"Starting async LinkedIn Premium Actor for {len(search_terms)} companies...")
    
    try:
        result = await client.call_actor_async(actor_id, run_input, timeout=cfg.get("timeout_seconds", 600))
        
        if result["status"] == "SUCCEEDED":
            logger.info(f"LinkedIn Premium Actor completed: {len(result['results'])} profiles found")
            return result["results"]
        else:
            logger.error(f"LinkedIn Premium Actor failed with status: {result['status']}")
            return []
            
    except Exception as e:
        logger.error(f"LinkedIn Premium Actor async execution failed: {e}")
        return []


async def _run_google_maps_contacts_async(places: List[Dict], client: AsyncApifyClient, cfg: dict) -> List[Dict]:
    """Async version of Google Maps Contact Details scraper."""
    if not places:
        return []
    
    actor_id = "lukaskrivka/google-maps-with-contact-details"
    
    # Extract place URLs
    place_urls = []
    for place in places:
        if place.get("placeUrl"):
            place_urls.append(place["placeUrl"])
    
    if not place_urls:
        return []
    
    # Limit based on configuration
    max_contacts = cfg.get("max_contact_enrichments", 50)
    place_urls = place_urls[:max_contacts]
    
    run_input = {
        "startUrls": [{"url": url} for url in place_urls],
        "language": "fr",
        "includeReviews": False,
        "maxReviews": 0,
        "includeImages": False,
    }
    
    logger.info(f"Starting async Google Maps Contacts for {len(place_urls)} places...")
    
    try:
        result = await client.call_actor_async(actor_id, run_input, timeout=cfg.get("timeout_seconds", 600))
        
        if result["status"] == "SUCCEEDED":
            logger.info(f"Google Maps Contacts completed: {len(result['results'])} contacts found")
            return result["results"]
        else:
            logger.error(f"Google Maps Contacts failed with status: {result['status']}")
            return []
            
    except Exception as e:
        logger.error(f"Google Maps Contacts async execution failed: {e}")
        return []


def _load_and_validate_input_async(cfg: dict, ctx: dict) -> Optional[pd.DataFrame]:
    """Load and validate input data for async processing."""
    # Load input data
    input_file = ctx.get("input_file")
    if not input_file:
        logger.error("No input file specified in context")
        return None
    
    try:
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        elif input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            logger.error(f"Unsupported file format: {input_file}")
            return None
        
        # Basic validation
        if df.empty:
            logger.error("Input file is empty")
            return None
        
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load input file {input_file}: {e}")
        return None


def _extract_addresses(df: pd.DataFrame) -> List[str]:
    """Extract addresses from dataframe."""
    addresses = []
    
    # Try different address column names
    address_columns = ['adresse', 'adresse_complete', 'address', 'full_address']
    
    for col in address_columns:
        if col in df.columns:
            address_series = df[col].dropna().astype(str)
            addresses = address_series.tolist()
            break
    
    # Filter out empty or placeholder addresses
    filtered_addresses = []
    for addr in addresses:
        if addr and addr.strip() and addr.upper() not in ['ADRESSE NON RENSEIGNEE', 'NOT PROVIDED']:
            filtered_addresses.append(addr.strip())
    
    logger.info(f"Extracted {len(filtered_addresses)} valid addresses")
    return filtered_addresses


def _extract_companies(df: pd.DataFrame) -> List[str]:
    """Extract company names from dataframe."""
    companies = []
    
    # Try different company name columns
    company_columns = ['denomination', 'company_name', 'raison_sociale', 'name']
    
    for col in company_columns:
        if col in df.columns:
            company_series = df[col].dropna().astype(str)
            companies = company_series.tolist()
            break
    
    # Filter out empty or placeholder companies
    filtered_companies = []
    for company in companies:
        if company and company.strip() and company.upper() not in ['DENOMINATION NON RENSEIGNEE', 'NOT PROVIDED']:
            filtered_companies.append(company.strip())
    
    logger.info(f"Extracted {len(filtered_companies)} valid company names")
    return filtered_companies


def _match_places_to_row(row: pd.Series, places: List[Dict]) -> List[Dict]:
    """Match places results to original row."""
    # Simple matching by address similarity
    # This could be enhanced with ML in the future
    matched = []
    row_address = str(row.get("adresse", "")).lower()
    
    for place in places:
        place_address = str(place.get("address", "")).lower()
        if any(word in place_address for word in row_address.split()[:3]):  # Match first 3 words
            matched.append(place)
    
    return matched


def _match_contacts_to_row(row: pd.Series, contacts: List[Dict]) -> List[Dict]:
    """Match contacts results to original row."""
    # Similar matching logic for contacts
    matched = []
    row_address = str(row.get("adresse", "")).lower()
    
    for contact in contacts:
        contact_address = str(contact.get("address", "")).lower()
        if any(word in contact_address for word in row_address.split()[:3]):
            matched.append(contact)
    
    return matched


def _match_linkedin_to_row(row: pd.Series, linkedin: List[Dict]) -> List[Dict]:
    """Match LinkedIn results to original row."""
    # Match by company name
    matched = []
    row_company = str(row.get("denomination", "")).lower()
    
    for profile in linkedin:
        profile_company = str(profile.get("company", "")).lower()
        if any(word in profile_company for word in row_company.split()[:2]):  # Match first 2 words
            matched.append(profile)
    
    return matched


def _combine_results_async(results: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """Combine async results with original data."""
    # Convert results to DataFrame format
    combined_data = []
    
    for result in results:
        row_data = result["original_data"].copy()
        
        # Add places data
        if result["places"]:
            place = result["places"][0]  # Take first match
            row_data.update({
                "google_place_id": place.get("placeId"),
                "google_rating": place.get("totalScore"),
                "google_reviews_count": place.get("reviewsCount"),
                "google_category": place.get("categoryName"),
            })
        
        # Add contact data
        if result["contacts"]:
            contact = result["contacts"][0]  # Take first match
            row_data.update({
                "phone_extracted": contact.get("phone"),
                "email_extracted": contact.get("email"),
                "website_extracted": contact.get("website"),
            })
        
        # Add LinkedIn data
        if result["linkedin"]:
            profile = result["linkedin"][0]  # Take first match
            row_data.update({
                "linkedin_profile": profile.get("profileUrl"),
                "executive_name": profile.get("fullName"),
                "executive_title": profile.get("title"),
            })
        
        combined_data.append(row_data)
    
    return pd.DataFrame(combined_data)


def _save_results_async(df: pd.DataFrame, cfg: dict, ctx: dict) -> str:
    """Save async results to file."""
    output_dir = ctx.get("output_dir", ".")
    output_file = f"{output_dir}/apify_enriched_async.parquet"
    
    try:
        df.to_parquet(output_file, index=False)
        logger.info(f"Async results saved to {output_file}")
        return output_file
    except Exception as e:
        # Fallback to CSV
        output_file = f"{output_dir}/apify_enriched_async.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Async results saved to {output_file} (CSV fallback)")
        return output_file


if __name__ == "__main__":
    # Example usage
    import sys
    import yaml
    
    if len(sys.argv) != 3:
        print("Usage: python apify_agents_async.py <config.yaml> <context.yaml>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    
    with open(sys.argv[2]) as f:
        ctx = yaml.safe_load(f)
    
    result = asyncio.run(run_async(cfg, ctx))
    print(json.dumps(result, indent=2))