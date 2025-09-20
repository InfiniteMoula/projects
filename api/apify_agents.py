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
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from apify_client import ApifyClient

from utils import io
from utils.parquet import ParquetBatchWriter
from utils.address_processor import AddressProcessor
from utils.company_name_processor import CompanyNameProcessor
from utils.quality_controller import GoogleMapsQualityController, LinkedInQualityController


def _get_apify_client() -> ApifyClient:
    """Get configured Apify client."""
    api_token = os.getenv('APIFY_API_TOKEN')
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
    
    # Calculate quality metrics
    result_stats = {
        "file": str(output_path),
        "status": "SUCCESS",
        "places_found": len(places_results),
        "contacts_enriched": len(contact_results),
        "linkedin_profiles": len(linkedin_results),
        "addresses_processed": len(df)
    }
    
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
    
    return result_stats