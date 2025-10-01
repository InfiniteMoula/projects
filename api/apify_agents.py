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
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
try:
    from apify_client import ApifyClient  # type: ignore
    APIFY_AVAILABLE = True
except ImportError:
    ApifyClient = Any  # type: ignore
    APIFY_AVAILABLE = False
from utils import io
from utils.parquet import ParquetBatchWriter
from utils.address_processor import AddressProcessor
from utils.company_name_processor import CompanyNameProcessor
from utils.quality_controller import GoogleMapsQualityController, LinkedInQualityController
from utils.retry_manager import LinkedInRetryManager, GoogleMapsRetryManager, RetryConfig
from utils.cost_manager import CostManager, ScraperType, create_cost_manager
from utils.dynamic_config import DynamicConfigurationManager, ConfigPriority, create_dynamic_config_manager
from utils.industry_optimizer import IndustryOptimizer, IndustryCategory, OptimizationStrategy, create_industry_optimizer

logger = logging.getLogger(__name__)


def _extract_priorities_from_config(cfg: dict, ctx: dict) -> List[ConfigPriority]:
    """Extract configuration priorities from config and context."""
    priorities = []
    
    # Check config priorities
    config_priorities = cfg.get("priorities", [])
    if isinstance(config_priorities, str):
        config_priorities = [config_priorities]
    
    for priority in config_priorities:
        try:
            priorities.append(ConfigPriority(priority.lower()))
        except ValueError:
            logger.warning(f"Unknown priority: {priority}")
    
    # Infer priorities from context
    if ctx.get("requires_contacts", False):
        priorities.append(ConfigPriority.CONTACTS)
    if ctx.get("requires_executives", False):
        priorities.append(ConfigPriority.EXECUTIVES)
    if ctx.get("location_critical", False):
        priorities.append(ConfigPriority.PLACES)
    if ctx.get("time_budget_min", 0) < 60:
        priorities.append(ConfigPriority.SPEED)
    
    # Default priorities if none specified
    if not priorities:
        priorities = [ConfigPriority.CONTACTS, ConfigPriority.QUALITY]
    
    return priorities


def _get_apify_client() -> "ApifyClient":
    """Get configured Apify client."""
    if not APIFY_AVAILABLE:
        raise RuntimeError("Apify integration is disabled (apify client library not installed)")

    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        raise ValueError("APIFY_API_TOKEN environment variable is required")

    return ApifyClient(api_token)


def _prepare_input_data(df: pd.DataFrame, cfg: dict) -> Dict:
    """
    Prepare and enhance input data for Apify agents.
    
    This function processes addresses and company names to optimize them
    for better search results from the Apify scrapers.
    """
    print("Preparing input data with address and company name optimization...")
    
    # Initialize processors
    address_processor = AddressProcessor()
    company_processor = CompanyNameProcessor()
    
    # Process addresses if available
    address_column = None
    for col in ['adresse', 'adresse_complete', 'address']:
        if col in df.columns:
            address_column = col
            break
    
    if address_column:
        print(f"Processing addresses from column: {address_column}")
        df_with_addresses = address_processor.process_dataframe(df, address_column)
        
        # Filter by address quality if configured
        min_quality = cfg.get('input_preparation', {}).get('min_address_quality', 0.3)
        high_quality_addresses = df_with_addresses[
            df_with_addresses['address_quality_score'] >= min_quality
        ]
        print(f"Filtered to {len(high_quality_addresses)} addresses with quality >= {min_quality}")
    else:
        print("No address column found")
        df_with_addresses = df
        high_quality_addresses = df
    
    # Process company names if available  
    company_column = None
    for col in ['denomination', 'raison_sociale', 'company_name', 'nom_entreprise']:
        if col in df.columns:
            company_column = col
            break
    
    if company_column:
        print(f"Processing company names from column: {company_column}")
        df_with_companies = company_processor.process_dataframe(high_quality_addresses, company_column)
        
        # Filter by company confidence if configured
        min_confidence = cfg.get('input_preparation', {}).get('min_company_confidence', 0.3)
        high_quality_companies = df_with_companies[
            df_with_companies['company_confidence'] >= min_confidence
        ]
        print(f"Filtered to {len(high_quality_companies)} companies with confidence >= {min_confidence}")
    else:
        print("No company name column found")
        df_with_companies = high_quality_addresses
        high_quality_companies = high_quality_addresses
    
    # Calculate overall input data quality score
    quality_scores = []
    for idx, row in high_quality_companies.iterrows():
        address_score = row.get('address_quality_score', 0.0)
        company_score = row.get('company_confidence', 0.0)
        
        # Weighted average (addresses are more important for Google Maps)
        overall_score = (address_score * 0.7) + (company_score * 0.3)
        quality_scores.append(overall_score)
    
    high_quality_companies = high_quality_companies.copy()
    high_quality_companies['input_quality_score'] = quality_scores
    
    # Sort by quality score (best first)
    high_quality_companies = high_quality_companies.sort_values(
        'input_quality_score', ascending=False
    )
    
    return {
        'processed_df': high_quality_companies,
        'address_processor': address_processor,
        'company_processor': company_processor,
        'total_records': len(df),
        'qualified_records': len(high_quality_companies),
        'quality_stats': {
            'mean_quality': float(high_quality_companies['input_quality_score'].mean()) if len(high_quality_companies) > 0 else 0.0,
            'min_quality': float(high_quality_companies['input_quality_score'].min()) if len(high_quality_companies) > 0 else 0.0,
            'max_quality': float(high_quality_companies['input_quality_score'].max()) if len(high_quality_companies) > 0 else 0.0
        }
    }


def _run_google_places_crawler_enhanced(
    addresses: List[str], 
    client: ApifyClient, 
    cfg: dict,
    cost_manager: CostManager,
    operation_id: str
) -> List[Dict]:
    """Enhanced Google Places Crawler with cost tracking and adaptive timeouts."""
    actor_id = "compass/crawler-google-places"
    
    # Apply adaptive timeout
    base_timeout = cfg.get("timeout_seconds", 300)
    timeout_seconds = base_timeout
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": addresses,
        "language": "fr",
        "maxCrawledPlaces": cfg.get("max_places_per_search", 10),
        "exportPlaceUrls": True,
        "includeImages": False,
        "includeReviews": False,
        "maxReviews": 0,
        "timeoutMs": timeout_seconds * 1000
    }
    
    logger.info(f"Starting Google Places Crawler for {len(addresses)} addresses (timeout: {timeout_seconds}s)")
    
    try:
        run = client.actor(actor_id).call(run_input=run_input)
        
        # Get results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        logger.info(f"Google Places Crawler completed: {len(results)} places found")
        return results
        
    except Exception as e:
        logger.error(f"Google Places Crawler failed: {e}")
        raise


def _run_google_maps_contact_details_enhanced(
    places_data: List[Dict], 
    client: ApifyClient, 
    cfg: dict,
    cost_manager: CostManager,
    operation_id: str
) -> List[Dict]:
    """Enhanced Google Maps Contact Details with cost optimization."""
    actor_id = "lukaskrivka/google-maps-with-contact-details"
    
    # Extract place URLs or search terms from places_data
    search_inputs = []
    for place in places_data:
        if 'url' in place:
            search_inputs.append(place['url'])
        elif 'title' in place and 'address' in place:
            search_inputs.append(f"{place['title']} {place['address']}")
    
    if not search_inputs:
        logger.warning("No valid inputs for Google Maps Contact Details enrichment")
        return []
    
    # Apply cost-aware limits
    max_enrichments = min(
        cfg.get("max_contact_enrichments", 50),
        len(search_inputs)
    )
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": search_inputs[:max_enrichments],
        "language": "fr",
        "maxCrawledPlaces": 1,
        "exportPlaceUrls": False,
        "timeoutMs": cfg.get("timeout_seconds", 300) * 1000
    }
    
    logger.info(f"Starting Google Maps Contact Details enrichment for {len(search_inputs[:max_enrichments])} places")
    
    try:
        run = client.actor(actor_id).call(run_input=run_input)
        
        # Get results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        logger.info(f"Google Maps Contact Details completed: {len(results)} places enriched")
        return results
        
    except Exception as e:
        logger.error(f"Google Maps Contact Details failed: {e}")
        raise


def _run_linkedin_premium_actor_enhanced(
    companies: List[str], 
    client: ApifyClient, 
    cfg: dict,
    cost_manager: CostManager,
    operation_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """Enhanced LinkedIn Premium Actor that returns both results and failed searches."""
    actor_id = "bebity/linkedin-premium-actor"
    
    # Prepare company search terms with cost limits
    max_searches = min(
        cfg.get("max_linkedin_searches", 20),
        len(companies)
    )
    search_terms = companies[:max_searches]
    
    if not search_terms:
        logger.warning("No company names for LinkedIn Premium Actor")
        return [], []
    
    # Prepare input for the actor
    run_input = {
        "searchTerms": search_terms,
        "language": "fr",
        "maxProfiles": cfg.get("max_profiles_per_company", 5),
        "filters": {
            "positions": cfg.get("filters", {}).get("positions", ["CEO", "CFO", "Director", "Founder", "Gérant", "Directeur", "Président"])
        },
        "timeoutMs": cfg.get("timeout_seconds", 600) * 1000
    }
    
    logger.info(f"Starting LinkedIn Premium Actor for {len(search_terms)} companies")
    
    try:
        run = client.actor(actor_id).call(run_input=run_input)
        
        # Get results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        # Identify failed searches (companies with no results)
        successful_companies = set()
        for result in results:
            if 'company' in result:
                successful_companies.add(result['company'])
        
        failed_searches = []
        for company in search_terms:
            if company not in successful_companies:
                failed_searches.append({
                    "company_name": company,
                    "searchTerms": [company],
                    "maxProfiles": run_input["maxProfiles"],
                    "filters": run_input["filters"]
                })
        
        logger.info(f"LinkedIn Premium Actor completed: {len(results)} profiles, {len(failed_searches)} failed companies")
        return results, failed_searches
        
    except Exception as e:
        logger.error(f"LinkedIn Premium Actor failed: {e}")
        # Return all companies as failed searches for retry
        failed_searches = []
        for company in search_terms:
            failed_searches.append({
                "company_name": company,
                "searchTerms": [company],
                "maxProfiles": run_input["maxProfiles"],
                "filters": run_input["filters"]
            })
        return [], failed_searches


def _run_linkedin_premium_search_single(
    search_input: Dict[str, Any], 
    client: ApifyClient, 
    cfg: dict
) -> List[Dict]:
    """Run a single LinkedIn Premium search for retry logic."""
    actor_id = "bebity/linkedin-premium-actor"
    
    # Use the modified search input from retry manager
    run_input = {
        "searchTerms": search_input.get("searchTerms", [search_input.get("company_name", "")]),
        "language": "fr",
        "maxProfiles": search_input.get("maxProfiles", 5),
        "filters": search_input.get("filters", {}),
        "timeoutMs": search_input.get("timeout", 600) * 1000
    }
    
    try:
        run = client.actor(actor_id).call(run_input=run_input)
        
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        return results
        
    except Exception as e:
        logger.error(f"Single LinkedIn search failed: {e}")
        return []


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


def _apply_quality_validation(df: pd.DataFrame, places_results: List[Dict], 
                            contact_results: List[Dict], linkedin_results: List[Dict],
                            quality_config: dict, outdir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply quality validation and confidence scoring to Apify results.
    
    Args:
        df: DataFrame with enriched results
        places_results: Raw Google Places results
        contact_results: Raw Google Maps contact results
        linkedin_results: Raw LinkedIn results
        quality_config: Quality control configuration
        outdir: Output directory for quality reports
        
    Returns:
        Tuple of (enhanced_df, quality_reports)
    """
    print("Applying quality validation to Apify results...")
    
    quality_reports = {}
    
    # Initialize quality controllers
    google_maps_controller = GoogleMapsQualityController(quality_config)
    linkedin_controller = LinkedInQualityController(quality_config)
    
    # Validate Google Maps results (places + contacts)
    if places_results or contact_results:
        combined_google_results = places_results + contact_results
        if combined_google_results:
            validated_google_results, google_report = google_maps_controller.validate_google_maps_batch(
                combined_google_results
            )
            quality_reports['google_maps'] = google_report
            
            # Export Google Maps quality dashboard data
            google_dashboard_path = outdir / "google_maps_quality_dashboard.json"
            google_maps_controller.export_quality_dashboard_data(google_report, str(google_dashboard_path))
            print(f"Google Maps quality report saved to {google_dashboard_path}")
    
    # Validate LinkedIn results
    if linkedin_results:
        validated_linkedin_results, linkedin_report = linkedin_controller.validate_linkedin_batch(
            linkedin_results
        )
        quality_reports['linkedin'] = linkedin_report
        
        # Export LinkedIn quality dashboard data
        linkedin_dashboard_path = outdir / "linkedin_quality_dashboard.json"
        linkedin_controller.export_quality_dashboard_data(linkedin_report, str(linkedin_dashboard_path))
        print(f"LinkedIn quality report saved to {linkedin_dashboard_path}")
    
    # Enhance DataFrame with quality scores
    df = _add_quality_scores_to_dataframe(df, places_results, contact_results, linkedin_results, 
                                         google_maps_controller, linkedin_controller)
    
    # Apply quality-based filtering if configured
    if quality_config.get("filter_low_quality", False):
        min_score = quality_config.get("min_quality_score", 0.5)
        original_count = len(df)
        df = df[df.get('overall_quality_score', 0.0) >= min_score]
        filtered_count = len(df)
        print(f"Quality filtering: kept {filtered_count}/{original_count} records (min score: {min_score})")
    
    # Generate combined quality summary
    combined_summary_path = outdir / "apify_quality_summary.json"
    _generate_combined_quality_summary(quality_reports, combined_summary_path)
    
    return df, quality_reports


def _add_quality_scores_to_dataframe(df: pd.DataFrame, places_results: List[Dict], 
                                   contact_results: List[Dict], linkedin_results: List[Dict],
                                   google_controller: GoogleMapsQualityController,
                                   linkedin_controller: LinkedInQualityController) -> pd.DataFrame:
    """Add quality scores and validation results to the main DataFrame."""
    
    # Initialize quality columns
    df['google_maps_quality_score'] = 0.0
    df['linkedin_quality_score'] = 0.0
    df['overall_quality_score'] = 0.0
    df['quality_confidence_level'] = 'low'
    df['quality_issues'] = ''
    df['quality_recommendations'] = ''
    
    # Process Google Maps quality scores
    if places_results or contact_results:
        google_results_map = {}
        
        # Map results by search terms or business names
        for result in places_results + contact_results:
            search_term = result.get('searchString', '')
            title = result.get('title', '')
            if search_term:
                google_results_map[search_term] = result
            elif title:
                google_results_map[title] = result
        
        # Match with DataFrame rows
        for idx, row in df.iterrows():
            address = row.get('adresse', '')
            business_name = row.get('apify_business_names', '').split(';')[0] if row.get('apify_business_names') else ''
            
            matching_result = None
            if address in google_results_map:
                matching_result = google_results_map[address]
            elif business_name and business_name in google_results_map:
                matching_result = google_results_map[business_name]
            
            if matching_result:
                validation = google_controller._validate_single_result(matching_result, 'google_maps_contacts')
                df.at[idx, 'google_maps_quality_score'] = validation.overall_score
                df.at[idx, 'quality_confidence_level'] = validation.confidence_level
                df.at[idx, 'quality_issues'] = '; '.join(validation.validation_issues[:3])  # Top 3 issues
                df.at[idx, 'quality_recommendations'] = '; '.join(validation.recommendations[:2])  # Top 2 recommendations
    
    # Process LinkedIn quality scores
    if linkedin_results:
        linkedin_results_map = {}
        
        # Map results by company names
        for result in linkedin_results:
            company_name = result.get('companyName', '')
            if company_name:
                if company_name not in linkedin_results_map:
                    linkedin_results_map[company_name] = []
                linkedin_results_map[company_name].append(result)
        
        # Match with DataFrame rows
        for idx, row in df.iterrows():
            company_names = []
            
            # Try multiple company name fields
            for field in ['company_name', 'denomination', 'raison_sociale']:
                if field in row and row[field]:
                    company_names.append(str(row[field]))
            
            if row.get('apify_business_names'):
                company_names.extend(row['apify_business_names'].split(';'))
            
            best_linkedin_score = 0.0
            for company_name in company_names:
                company_name = company_name.strip()
                if company_name in linkedin_results_map:
                    # Get best profile for this company
                    company_profiles = linkedin_results_map[company_name]
                    for profile in company_profiles:
                        validation = linkedin_controller._validate_single_result(profile, 'linkedin_premium')
                        if validation.overall_score > best_linkedin_score:
                            best_linkedin_score = validation.overall_score
            
            df.at[idx, 'linkedin_quality_score'] = best_linkedin_score
    
    # Calculate overall quality score
    for idx, row in df.iterrows():
        google_score = row.get('google_maps_quality_score', 0.0)
        linkedin_score = row.get('linkedin_quality_score', 0.0)
        
        # Weighted average (Google Maps weighted higher for contact info)
        overall_score = (google_score * 0.7) + (linkedin_score * 0.3)
        df.at[idx, 'overall_quality_score'] = overall_score
        
        # Update confidence level based on overall score
        if overall_score >= 0.8:
            df.at[idx, 'quality_confidence_level'] = 'high'
        elif overall_score >= 0.6:
            df.at[idx, 'quality_confidence_level'] = 'medium'
        else:
            df.at[idx, 'quality_confidence_level'] = 'low'
    
    return df


def _generate_combined_quality_summary(quality_reports: Dict, output_path: Path) -> None:
    """Generate a combined quality summary from all sources."""
    
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'sources': list(quality_reports.keys()),
        'overall_stats': {},
        'source_details': {}
    }
    
    total_records = 0
    total_valid = 0
    all_scores = []
    
    for source, report in quality_reports.items():
        total_records += report.total_records
        total_valid += report.valid_records
        all_scores.extend([report.average_score] * report.total_records)
        
        summary['source_details'][source] = {
            'total_records': report.total_records,
            'valid_records': report.valid_records,
            'validation_rate': round(report.validation_rate * 100, 1),
            'average_score': round(report.average_score * 100, 1),
            'confidence_distribution': report.confidence_distribution,
            'top_issues': report.common_issues[:3]
        }
    
    # Calculate overall statistics
    summary['overall_stats'] = {
        'total_records': total_records,
        'total_valid': total_valid,
        'overall_validation_rate': round((total_valid / total_records * 100) if total_records > 0 else 0, 1),
        'overall_average_score': round((sum(all_scores) / len(all_scores) * 100) if all_scores else 0, 1)
    }
    
    # Generate overall recommendations
    recommendations = []
    if total_records > 0:
        validation_rate = total_valid / total_records
        if validation_rate < 0.8:
            recommendations.append("Consider improving data source quality")
        if len(all_scores) > 0 and sum(all_scores) / len(all_scores) < 0.6:
            recommendations.append("Review extraction and validation methods")
    
    summary['recommendations'] = recommendations
    
    # Save summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Combined quality summary saved to {output_path}")


def run(cfg: dict, ctx: dict) -> dict:
    """
    Run Apify agents to enrich business data with intelligent automation.
    
    This enhanced version includes:
    - Smart retry logic for failed searches
    - Dynamic configuration based on budget and priorities  
    - Industry-specific optimization
    - Real-time cost monitoring
    """
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    io.ensure_dir(outdir)
    
    # Initialize intelligent automation components
    cost_manager = create_cost_manager(cfg)
    dynamic_config = create_dynamic_config_manager(cost_manager)
    industry_optimizer = create_industry_optimizer()
    
    logger.info("Intelligent Automation initialized")
    
    # Check for dry run
    if ctx.get("dry_run"):
        empty_path = outdir / "apify_empty.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "DRY_RUN"}
    
    # Extract budget and priorities from context
    available_budget = ctx.get("available_budget", cfg.get("budget", {}).get("total_daily_budget", 1000.0))
    priorities = _extract_priorities_from_config(cfg, ctx)
    
    # Apply dynamic configuration optimization
    try:
        optimized_config = dynamic_config.determine_optimal_configuration(
            base_config=cfg,
            available_budget=available_budget,
            priorities=priorities,
            context=ctx
        )
        apify_config = optimized_config.get("apify", {})
        logger.info(f"Dynamic configuration applied - budget: {available_budget}, priorities: {[p.value for p in priorities]}")
    except Exception as e:
        logger.warning(f"Dynamic configuration failed, using base config: {e}")
        apify_config = cfg.get("apify", {})
    
    # Check if Apify is configured
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
    
    # Prepare and enhance input data
    try:
        input_data = _prepare_input_data(df, cfg)
        processed_df = input_data['processed_df']
        
        print(f"Input preparation completed:")
        print(f"  - Total records: {input_data['total_records']}")
        print(f"  - Qualified records: {input_data['qualified_records']}")
        print(f"  - Quality stats: {input_data['quality_stats']}")
        
        # Use processed data for scraping
        if processed_df.empty:
            empty_path = outdir / "apify_no_qualified_data.parquet"
            pd.DataFrame().to_parquet(empty_path)
            return {"file": str(empty_path), "status": "NO_QUALIFIED_DATA"}
        
        # Update df to use processed version
        df = processed_df
        
    except Exception as e:
        print(f"Warning: Input preparation failed: {e}")
        print("Proceeding with original data...")
    
    # Detect industry and apply optimizations
    detected_industry = IndustryCategory.OTHER
    optimization_strategy = OptimizationStrategy.CONTACT_FOCUSED
    
    if not df.empty:
        # Try to detect industry from first few records
        sample_data = {}
        if 'naf_code' in df.columns:
            sample_data['naf_code'] = df['naf_code'].iloc[0] if not df['naf_code'].isna().all() else ""
        
        company_col = None
        for col in ['company_name', 'denomination', 'raison_sociale', 'company_primary']:
            if col in df.columns:
                company_col = col
                break
        
        if company_col:
            sample_data['company_name'] = df[company_col].iloc[0] if not df[company_col].isna().all() else ""
        
        try:
            detected_industry, confidence = industry_optimizer.detect_industry(sample_data)
            logger.info(f"Detected industry: {detected_industry.value} (confidence: {confidence:.2f})")
            
            # Get optimization strategy suggestion
            business_context = {
                "requires_contacts": ConfigPriority.CONTACTS in priorities,
                "requires_executives": ConfigPriority.EXECUTIVES in priorities,
                "location_critical": ConfigPriority.PLACES in priorities,
                "comprehensive_data": ConfigPriority.QUALITY in priorities
            }
            
            optimization_strategy, strategy_reason = industry_optimizer.suggest_optimization_strategy(
                detected_industry, business_context, available_budget
            )
            logger.info(f"Selected optimization strategy: {optimization_strategy.value} - {strategy_reason}")
            
            # Apply industry-specific optimizations
            apify_config = industry_optimizer.optimize_configuration(
                {"apify": apify_config}, detected_industry, optimization_strategy, ctx
            )["apify"]
            
        except Exception as e:
            logger.warning(f"Industry optimization failed: {e}")
    
    # Extract addresses and company names (use enhanced versions if available)
    address_column = 'address_primary' if 'address_primary' in df.columns else 'adresse'
    company_column = 'company_primary' if 'company_primary' in df.columns else None
    
    if company_column is None:
        for col in ['company_name', 'denomination', 'raison_sociale']:
            if col in df.columns:
                company_column = col
                break
    
    addresses = df[address_column].fillna('').tolist() if address_column in df.columns else []
    company_names = df[company_column].fillna('').tolist() if company_column else []
    
    # Initialize Apify client
    try:
        client = _get_apify_client()
    except ValueError as e:
        empty_path = outdir / "apify_client_error.parquet"
        pd.DataFrame().to_parquet(empty_path)
        return {"file": str(empty_path), "status": "CLIENT_ERROR", "error": str(e)}
    
    # Initialize retry managers
    retry_config = RetryConfig(
        max_retries=apify_config.get("retry_settings", {}).get("max_retries", 3),
        max_cost_per_search=apify_config.get("retry_settings", {}).get("max_cost_per_search", 150.0),
        strategies=apify_config.get("retry_settings", {}).get("strategies", ["simplified_name", "alternative_positions", "broader_search"])
    )
    
    linkedin_retry_manager = LinkedInRetryManager(retry_config)
    google_maps_retry_manager = GoogleMapsRetryManager(retry_config)
    
    # Initialize results and tracking
    places_results = []
    contact_results = []
    linkedin_results = []
    failed_operations = {"places": [], "contacts": [], "linkedin": []}
    
    # Enhanced Google Places Crawler execution with cost tracking
    if apify_config.get("google_places", {}).get("enabled", True):
        operation_id = f"places_{uuid.uuid4().hex[:8]}"
        operation_details = {
            "addresses_count": len(addresses),
            "max_places_per_search": apify_config.get("google_places", {}).get("max_places_per_search", 10)
        }
        
        # Pre-operation cost check
        cost_check = cost_manager.pre_operation_check(ScraperType.GOOGLE_PLACES, operation_details)
        
        if cost_check["can_proceed"]:
            # Track operation start
            cost_entry = cost_manager.track_operation_start(
                ScraperType.GOOGLE_PLACES, operation_id, operation_details
            )
            
            try:
                places_results = _run_google_places_crawler_enhanced(
                    addresses, client, apify_config.get("google_places", {}), cost_manager, operation_id
                )
                
                # Track successful completion
                cost_manager.track_operation_complete(
                    operation_id, success=True, results_count=len(places_results)
                )
                
                logger.info(f"Google Places completed successfully: {len(places_results)} results")
                
            except Exception as e:
                logger.error(f"Google Places operation failed: {e}")
                cost_manager.track_operation_complete(operation_id, success=False)
                failed_operations["places"].append({"addresses": addresses, "error": str(e)})
        else:
            logger.warning(f"Google Places operation blocked: {cost_check['blocks']}")
    
    # Enhanced Google Maps Contact Details with retry logic
    if apify_config.get("google_maps_contacts", {}).get("enabled", True) and places_results:
        operation_id = f"contacts_{uuid.uuid4().hex[:8]}"
        operation_details = {
            "places_count": len(places_results),
            "max_contact_enrichments": apify_config.get("google_maps_contacts", {}).get("max_contact_enrichments", 50)
        }
        
        cost_check = cost_manager.pre_operation_check(ScraperType.GOOGLE_MAPS_CONTACTS, operation_details)
        
        if cost_check["can_proceed"]:
            cost_entry = cost_manager.track_operation_start(
                ScraperType.GOOGLE_MAPS_CONTACTS, operation_id, operation_details
            )
            
            try:
                contact_results = _run_google_maps_contact_details_enhanced(
                    places_results, client, apify_config.get("google_maps_contacts", {}), cost_manager, operation_id
                )
                
                cost_manager.track_operation_complete(
                    operation_id, success=True, results_count=len(contact_results)
                )
                
                logger.info(f"Google Maps Contacts completed: {len(contact_results)} results")
                
            except Exception as e:
                logger.error(f"Google Maps Contacts operation failed: {e}")
                cost_manager.track_operation_complete(operation_id, success=False)
                failed_operations["contacts"].append({"places": places_results, "error": str(e)})
                
                # Attempt retry with fallback strategy
                try:
                    logger.info("Attempting Google Maps retry with fallback strategy")
                    contact_results = google_maps_retry_manager.retry_failed_searches(
                        [{"places": places_results, "error": str(e)}],
                        client,
                        lambda modified_input, client: _run_google_maps_contact_details(
                            modified_input["places"], client, apify_config.get("google_maps_contacts", {})
                        )
                    )
                    logger.info(f"Google Maps retry recovered {len(contact_results)} results")
                except Exception as retry_error:
                    logger.error(f"Google Maps retry also failed: {retry_error}")
        else:
            logger.warning(f"Google Maps Contacts operation blocked: {cost_check['blocks']}")
    
    # Enhanced LinkedIn Premium Actor with intelligent retry
    valid_companies = [name for name in company_names if name and name.strip()]
    if apify_config.get("linkedin_premium", {}).get("enabled", True) and valid_companies:
        operation_id = f"linkedin_{uuid.uuid4().hex[:8]}"
        operation_details = {
            "companies_count": len(valid_companies),
            "max_linkedin_searches": apify_config.get("linkedin_premium", {}).get("max_linkedin_searches", 20),
            "max_profiles_per_company": apify_config.get("linkedin_premium", {}).get("max_profiles_per_company", 5)
        }
        
        cost_check = cost_manager.pre_operation_check(ScraperType.LINKEDIN_PREMIUM, operation_details)
        
        if cost_check["can_proceed"]:
            cost_entry = cost_manager.track_operation_start(
                ScraperType.LINKEDIN_PREMIUM, operation_id, operation_details
            )
            
            try:
                linkedin_results, failed_searches = _run_linkedin_premium_actor_enhanced(
                    valid_companies, client, apify_config.get("linkedin_premium", {}), cost_manager, operation_id
                )
                
                cost_manager.track_operation_complete(
                    operation_id, success=True, results_count=len(linkedin_results)
                )
                
                logger.info(f"LinkedIn Premium completed: {len(linkedin_results)} results, {len(failed_searches)} failed")
                
                # Attempt intelligent retry for failed searches
                if failed_searches and linkedin_retry_manager:
                    try:
                        logger.info(f"Attempting intelligent retry for {len(failed_searches)} failed LinkedIn searches")
                        
                        retry_results = linkedin_retry_manager.retry_failed_searches(
                            failed_searches,
                            client,
                            lambda modified_input, client: _run_linkedin_premium_search_single(
                                modified_input, client, apify_config.get("linkedin_premium", {})
                            )
                        )
                        
                        linkedin_results.extend(retry_results)
                        retry_stats = linkedin_retry_manager.get_retry_stats()
                        logger.info(f"LinkedIn retry completed: {len(retry_results)} recovered, stats: {retry_stats}")
                        
                    except Exception as retry_error:
                        logger.error(f"LinkedIn retry failed: {retry_error}")
                
            except Exception as e:
                logger.error(f"LinkedIn Premium operation failed: {e}")
                cost_manager.track_operation_complete(operation_id, success=False)
                failed_operations["linkedin"].append({"companies": valid_companies, "error": str(e)})
        else:
            logger.warning(f"LinkedIn Premium operation blocked: {cost_check['blocks']}")
    
    # Check budget thresholds and generate alerts
    budget_alerts = cost_manager.check_budget_thresholds()
    if budget_alerts:
        logger.warning(f"Budget alerts generated: {len(budget_alerts)}")
        for alert in budget_alerts:
            logger.warning(f"Budget Alert: {alert.message}")
    
    # Merge results
    enriched_df = _merge_apify_results(df, places_results, contact_results, linkedin_results)
    
    # Apply quality control and validation
    quality_config = apify_config.get("quality_control", {})
    if quality_config.get("enabled", True):
        enriched_df, quality_reports = _apply_quality_validation(
            enriched_df, places_results, contact_results, linkedin_results, quality_config, outdir
        )
    
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
    
    # Save intelligent automation reports
    automation_reports = {}
    
    # Cost management report
    cost_summary = cost_manager.get_cost_summary()
    automation_reports["cost_management"] = cost_summary
    
    # Retry statistics
    if linkedin_retry_manager:
        retry_stats = linkedin_retry_manager.get_retry_stats()
        automation_reports["linkedin_retry_stats"] = retry_stats
    
    # Dynamic configuration report
    config_summary = dynamic_config.get_configuration_summary()
    automation_reports["dynamic_configuration"] = config_summary
    
    # Industry optimization insights
    industry_insights = industry_optimizer.get_industry_insights(detected_industry)
    automation_reports["industry_optimization"] = {
        "detected_industry": detected_industry.value,
        "optimization_strategy": optimization_strategy.value,
        "insights": industry_insights
    }
    
    # Save automation reports
    automation_path = outdir / "intelligent_automation_report.json"
    io.write_json(automation_path, automation_reports)
    
    # Calculate quality metrics
    result_stats = {
        "file": str(output_path),
        "status": "SUCCESS",
        "places_found": len(places_results),
        "contacts_enriched": len(contact_results),
        "linkedin_profiles": len(linkedin_results),
        "addresses_processed": len(df)
    }
    
    # Add intelligent automation metrics
    real_time_costs = cost_manager.get_real_time_costs()
    result_stats.update({
        "intelligent_automation": {
            "detected_industry": detected_industry.value,
            "optimization_strategy": optimization_strategy.value,
            "total_cost": real_time_costs["total_current_cost"],
            "budget_usage_pct": real_time_costs["daily_budget_usage_pct"],
            "cost_efficiency_score": real_time_costs["cost_efficiency_score"],
            "retry_attempts": linkedin_retry_manager.get_retry_stats()["total_attempts"] if linkedin_retry_manager else 0,
            "successful_retries": linkedin_retry_manager.get_retry_stats()["successful_attempts"] if linkedin_retry_manager else 0,
            "budget_alerts": len(budget_alerts),
            "failed_operations": {
                "places": len(failed_operations["places"]),
                "contacts": len(failed_operations["contacts"]),
                "linkedin": len(failed_operations["linkedin"])
            }
        },
        "automation_report_file": str(automation_path)
    })
    
    # Add input quality stats if available
    if 'input_data' in locals():
        result_stats.update({
            "input_quality_stats": input_data['quality_stats'],
            "total_input_records": input_data['total_records'],
            "qualified_input_records": input_data['qualified_records'],
            "qualification_rate": input_data['qualified_records'] / input_data['total_records'] if input_data['total_records'] > 0 else 0.0
        })
    
    # Add quality validation stats if available
    if 'quality_reports' in locals():
        overall_quality_stats = {}
        for source, report in quality_reports.items():
            overall_quality_stats[f"{source}_validation_rate"] = round(report.validation_rate * 100, 1)
            overall_quality_stats[f"{source}_average_score"] = round(report.average_score * 100, 1)
        result_stats["quality_validation_stats"] = overall_quality_stats
    
    # Log final summary
    logger.info("=== Intelligent Automation Summary ===")
    logger.info(f"Industry: {detected_industry.value}")
    logger.info(f"Strategy: {optimization_strategy.value}")
    logger.info(f"Total Cost: {real_time_costs['total_current_cost']:.1f} credits")
    logger.info(f"Budget Usage: {real_time_costs['daily_budget_usage_pct']:.1f}%")
    logger.info(f"Efficiency Score: {real_time_costs['cost_efficiency_score']:.1f}")
    if linkedin_retry_manager:
        retry_stats = linkedin_retry_manager.get_retry_stats()
        logger.info(f"Retry Success Rate: {retry_stats.get('success_rate', 0):.1f}%")
    logger.info("=====================================")
    
    return result_stats
