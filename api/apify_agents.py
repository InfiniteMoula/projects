#!/usr/bin/env python3
"""
Apify API agents for business data enrichment.

This module provides integration with three Apify scrapers:
1. Google Places Crawler (compass/crawler-google-places)
2. Google Maps with Contact Details (lukaskrivka/google-maps-with-contact-details)
3. LinkedIn Premium Actor (bebity/linkedin-premium-actor)

The module reads addresses from step 7 (address extraction) and enriches business data
using the Apify platform scrapers.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from apify_client import ApifyClient

from utils import io
from utils.parquet import ParquetBatchWriter


def _get_apify_client() -> ApifyClient:
    """Get configured Apify client."""
    api_token = os.getenv('APIFY_API_TOKEN')
    if not api_token:
        raise ValueError("APIFY_API_TOKEN environment variable is required")
    
    return ApifyClient(api_token)


def _run_google_places_crawler(addresses: List[str], client: ApifyClient, cfg: dict) -> List[Dict]:
    """
    Run the Google Places Crawler (compass/crawler-google-places).
    
    This scraper searches for addresses and gets all business information.
    """
    actor_id = "compass/crawler-google-places"
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": addresses,
        "language": "fr",
        "maxCrawledPlaces": cfg.get("max_places_per_search", 10),
        "exportPlaceUrls": True,
        "includeImages": False,
        "includeReviews": False,
        "maxReviews": 0,
    }
    
    # Run the actor
    print(f"Starting Google Places Crawler for {len(addresses)} addresses...")
    run = client.actor(actor_id).call(run_input=run_input)
    
    # Get results
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)
    
    print(f"Google Places Crawler completed. Found {len(results)} places.")
    return results


def _run_google_maps_contact_details(places_data: List[Dict], client: ApifyClient, cfg: dict) -> List[Dict]:
    """
    Run Google Maps with Contact Details (lukaskrivka/google-maps-with-contact-details).
    
    This enriches the places data with contact details.
    """
    actor_id = "lukaskrivka/google-maps-with-contact-details"
    
    # Extract place URLs or search terms from places_data
    search_inputs = []
    for place in places_data:
        if 'url' in place:
            search_inputs.append(place['url'])
        elif 'title' in place and 'address' in place:
            search_inputs.append(f"{place['title']} {place['address']}")
    
    if not search_inputs:
        print("No valid inputs for Google Maps Contact Details enrichment")
        return []
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": search_inputs[:cfg.get("max_contact_enrichments", 50)],
        "language": "fr",
        "maxCrawledPlaces": 1,  # We already have the places, just need contact details
        "exportPlaceUrls": False,
    }
    
    # Run the actor
    print(f"Starting Google Maps Contact Details enrichment for {len(search_inputs)} places...")
    run = client.actor(actor_id).call(run_input=run_input)
    
    # Get results
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)
    
    print(f"Google Maps Contact Details completed. Enriched {len(results)} places.")
    return results


def _run_linkedin_premium_actor(companies: List[str], client: ApifyClient, cfg: dict) -> List[Dict]:
    """
    Run LinkedIn Premium Actor (bebity/linkedin-premium-actor).
    
    This gets CEO, CFO, Directors and Founder information.
    """
    actor_id = "bebity/linkedin-premium-actor"
    
    # Prepare company search terms
    search_terms = companies[:cfg.get("max_linkedin_searches", 20)]
    
    if not search_terms:
        print("No company names for LinkedIn Premium Actor")
        return []
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": search_terms,
        "language": "fr",
        "maxProfiles": cfg.get("max_profiles_per_company", 5),
        "filters": {
            "positions": ["CEO", "CFO", "Director", "Founder", "Gérant", "Directeur", "Président"]
        }
    }
    
    # Run the actor
    print(f"Starting LinkedIn Premium Actor for {len(search_terms)} companies...")
    run = client.actor(actor_id).call(run_input=run_input)
    
    # Get results
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)
    
    print(f"LinkedIn Premium Actor completed. Found {len(results)} profiles.")
    return results


def _merge_apify_results(addresses_df: pd.DataFrame, 
                        places_results: List[Dict], 
                        contact_results: List[Dict], 
                        linkedin_results: List[Dict]) -> pd.DataFrame:
    """
    Merge Apify results back with the original addresses DataFrame.
    """
    # Create enrichment columns
    enriched_df = addresses_df.copy()
    
    # Initialize new columns
    enriched_df['apify_places_found'] = 0
    enriched_df['apify_business_names'] = ''
    enriched_df['apify_phones'] = ''
    enriched_df['apify_emails'] = ''
    enriched_df['apify_websites'] = ''
    enriched_df['apify_ratings'] = ''
    enriched_df['apify_categories'] = ''
    enriched_df['apify_executives'] = ''
    enriched_df['apify_linkedin_profiles'] = ''
    
    # Process places results
    places_by_search = {}
    for place in places_results:
        search_term = place.get('searchString', '')
        if search_term not in places_by_search:
            places_by_search[search_term] = []
        places_by_search[search_term].append(place)
    
    # Process contact results
    contact_by_place = {}
    for contact in contact_results:
        place_name = contact.get('title', '')
        contact_by_place[place_name] = contact
    
    # Process LinkedIn results
    linkedin_by_company = {}
    for profile in linkedin_results:
        company_name = profile.get('companyName', '')
        if company_name not in linkedin_by_company:
            linkedin_by_company[company_name] = []
        linkedin_by_company[company_name].append(profile)
    
    # Merge results into dataframe
    for idx, row in enriched_df.iterrows():
        address = row['adresse']
        company_name = row.get('company_name', '')
        
        # Find matching places
        matching_places = places_by_search.get(address, [])
        if matching_places:
            enriched_df.at[idx, 'apify_places_found'] = len(matching_places)
            
            # Extract business information
            business_names = [p.get('title', '') for p in matching_places if p.get('title')]
            phones = []
            emails = []
            websites = []
            ratings = []
            categories = []
            
            for place in matching_places:
                # Phone numbers
                if place.get('phone'):
                    phones.append(place['phone'])
                
                # Emails
                if place.get('email'):
                    emails.append(place['email'])
                
                # Websites
                if place.get('website'):
                    websites.append(place['website'])
                
                # Ratings
                if place.get('totalScore'):
                    ratings.append(str(place['totalScore']))
                
                # Categories
                if place.get('categoryName'):
                    categories.append(place['categoryName'])
                
                # Check for contact enrichment
                place_title = place.get('title', '')
                if place_title in contact_by_place:
                    contact_info = contact_by_place[place_title]
                    if contact_info.get('phone') and contact_info['phone'] not in phones:
                        phones.append(contact_info['phone'])
                    if contact_info.get('email') and contact_info['email'] not in emails:
                        emails.append(contact_info['email'])
                    if contact_info.get('website') and contact_info['website'] not in websites:
                        websites.append(contact_info['website'])
            
            # Store aggregated data
            enriched_df.at[idx, 'apify_business_names'] = '; '.join(business_names[:3])
            enriched_df.at[idx, 'apify_phones'] = '; '.join(phones[:3])
            enriched_df.at[idx, 'apify_emails'] = '; '.join(emails[:3])
            enriched_df.at[idx, 'apify_websites'] = '; '.join(websites[:3])
            enriched_df.at[idx, 'apify_ratings'] = '; '.join(ratings[:3])
            enriched_df.at[idx, 'apify_categories'] = '; '.join(categories[:3])
        
        # Find matching LinkedIn profiles
        if company_name and company_name in linkedin_by_company:
            profiles = linkedin_by_company[company_name]
            executive_info = []
            linkedin_urls = []
            
            for profile in profiles:
                name = profile.get('fullName', '')
                position = profile.get('position', '')
                linkedin_url = profile.get('profileUrl', '')
                
                if name and position:
                    executive_info.append(f"{name} ({position})")
                
                if linkedin_url:
                    linkedin_urls.append(linkedin_url)
            
            enriched_df.at[idx, 'apify_executives'] = '; '.join(executive_info[:5])
            enriched_df.at[idx, 'apify_linkedin_profiles'] = '; '.join(linkedin_urls[:5])
    
    return enriched_df


def run(cfg: dict, ctx: dict) -> dict:
    """
    Run Apify agents to enrich business data.
    
    This step requires addresses from step 7 (address extraction) and uses
    three Apify scrapers to gather comprehensive business information.
    """
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    io.ensure_dir(outdir)
    
    # Check for dry run
    if ctx.get("dry_run"):
        empty_path = outdir / "apify_empty.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "DRY_RUN"}
    
    # Check if Apify is configured
    apify_config = cfg.get("apify", {})
    if not apify_config.get("enabled", False):
        empty_path = outdir / "apify_disabled.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "DISABLED"}
    
    # Check for API token
    if not os.getenv('APIFY_API_TOKEN'):
        empty_path = outdir / "apify_no_token.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "NO_TOKEN", 
                "error": "APIFY_API_TOKEN environment variable is required"}
    
    # Look for addresses from step 7
    database_path = outdir / "database.csv"
    if not database_path.exists():
        # Fallback to normalized data
        normalized_path = outdir / "normalized.parquet"
        if not normalized_path.exists():
            empty_path = outdir / "apify_no_input.parquet"
            pd.DataFrame().to_parquet(empty_path)
            return {"file": str(empty_path), "status": "NO_INPUT", 
                    "error": "No address data found (database.csv or normalized.parquet)"}
        
        # Read from normalized data
        df = pd.read_parquet(normalized_path)
        if 'adresse' not in df.columns:
            empty_path = outdir / "apify_no_addresses.parquet"
            pd.DataFrame().to_parquet(empty_path)
            return {"file": str(empty_path), "status": "NO_ADDRESSES", 
                    "error": "No 'adresse' column found in normalized data"}
    else:
        # Read from database.csv (step 7 output)
        df = pd.read_csv(database_path)
    
    # Filter addresses
    if 'adresse' not in df.columns:
        empty_path = outdir / "apify_no_addresses.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "NO_ADDRESSES", 
                "error": "No 'adresse' column found"}
    
    df = df[df['adresse'].notna() & (df['adresse'].str.strip() != '')]
    
    if df.empty:
        empty_path = outdir / "apify_empty_addresses.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "EMPTY_ADDRESSES"}
    
    # Limit addresses for testing/cost control
    max_addresses = apify_config.get("max_addresses", 10)
    if len(df) > max_addresses:
        df = df.head(max_addresses)
        print(f"Limited to {max_addresses} addresses for Apify processing")
    
    # Extract addresses and company names
    addresses = df['adresse'].tolist()
    company_names = df.get('company_name', df.get('denomination', df.get('raison_sociale', pd.Series(dtype=str)))).fillna('').tolist()
    
    # Initialize Apify client
    try:
        client = _get_apify_client()
    except ValueError as e:
        empty_path = outdir / "apify_client_error.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "CLIENT_ERROR", "error": str(e)}
    
    # Initialize results
    places_results = []
    contact_results = []
    linkedin_results = []
    
    # Run Google Places Crawler if enabled
    if apify_config.get("google_places", {}).get("enabled", True):
        try:
            places_results = _run_google_places_crawler(addresses, client, apify_config.get("google_places", {}))
        except Exception as e:
            print(f"Error running Google Places Crawler: {e}")
    
    # Run Google Maps Contact Details if enabled and we have places
    if apify_config.get("google_maps_contacts", {}).get("enabled", True) and places_results:
        try:
            contact_results = _run_google_maps_contact_details(places_results, client, apify_config.get("google_maps_contacts", {}))
        except Exception as e:
            print(f"Error running Google Maps Contact Details: {e}")
    
    # Run LinkedIn Premium Actor if enabled and we have company names
    valid_companies = [name for name in company_names if name and name.strip()]
    if apify_config.get("linkedin_premium", {}).get("enabled", True) and valid_companies:
        try:
            linkedin_results = _run_linkedin_premium_actor(valid_companies, client, apify_config.get("linkedin_premium", {}))
        except Exception as e:
            print(f"Error running LinkedIn Premium Actor: {e}")
    
    # Merge results
    enriched_df = _merge_apify_results(df, places_results, contact_results, linkedin_results)
    
    # Save results
    output_path = outdir / "apify_enriched.parquet"
    enriched_df.to_parquet(output_path)
    
    # Save raw results for debugging
    if apify_config.get("save_raw_results", True):
        raw_results = {
            "places": places_results,
            "contacts": contact_results,
            "linkedin": linkedin_results
        }
        raw_path = outdir / "apify_raw_results.json"
        io.write_json(raw_path, raw_results)
    
    return {
        "file": str(output_path),
        "status": "SUCCESS",
        "places_found": len(places_results),
        "contacts_enriched": len(contact_results),
        "linkedin_profiles": len(linkedin_results),
        "addresses_processed": len(df)
    }