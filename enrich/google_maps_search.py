#!/usr/bin/env python3
"""
Google Maps search enrichment module.

Searches maps.google.com for business information using address components to extract:
- Business names and types
- Phone numbers
- Reviews and ratings
- Business hours
- Additional contact information
- Services offered

Results are merged into the dataset to enrich company information.
"""

import re
import time
import urllib.parse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import httpx
from bs4 import BeautifulSoup

from utils.parquet import ParquetBatchWriter

# Configuration
MAPS_TIMEOUT = 15.0
REQUEST_DELAY = (2.0, 4.0)  # Random delay between requests (min, max) in seconds
MAX_WORKERS = 1  # Very limited to avoid being blocked by Google Maps
RETRY_COUNT = 2

# User agent that looks like a regular browser
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

# Regular expressions for extracting information
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
RATING_REGEX = re.compile(r'(\d+[.,]\d+)\s*(?:étoiles?|stars?|★)', re.IGNORECASE)
REVIEWS_REGEX = re.compile(r'(\d+)\s*(?:avis|reviews?|commentaires?)', re.IGNORECASE)

def _build_address_query(row: pd.Series) -> str:
    """Build address query from DataFrame row."""
    # Try to build from individual components if available
    address_parts = []
    
    # Check for individual address components first
    if 'numero_voie' in row and pd.notna(row['numero_voie']):
        address_parts.append(str(row['numero_voie']))
    if 'type_voie' in row and pd.notna(row['type_voie']):
        address_parts.append(str(row['type_voie']))
    if 'libelle_voie' in row and pd.notna(row['libelle_voie']):
        address_parts.append(str(row['libelle_voie']))
    
    # If individual components are not available, use complete address
    if not address_parts and 'adresse' in row and pd.notna(row['adresse']):
        address_parts.append(str(row['adresse']))
    
    # Add city and postal code
    if 'commune' in row and pd.notna(row['commune']):
        address_parts.append(str(row['commune']))
    if 'code_postal' in row and pd.notna(row['code_postal']):
        address_parts.append(str(row['code_postal']))
    
    # Add company name for better results
    if 'raison_sociale' in row and pd.notna(row['raison_sociale']):
        company_name = str(row['raison_sociale'])
        # Place company name at the beginning for better search results
        address_parts.insert(0, company_name)
    
    return " ".join(address_parts).strip()

def _build_google_maps_url(query: str) -> str:
    """Build Google Maps search URL."""
    encoded_query = urllib.parse.quote_plus(query)
    return f"https://maps.google.com/maps/search/{encoded_query}"

def _extract_business_info_from_maps(html_content: str) -> Dict[str, any]:
    """Extract business information from Google Maps HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = {
        'business_names': [],
        'phone_numbers': [],
        'emails': [],
        'ratings': [],
        'review_counts': [],
        'business_types': [],
        'hours': [],
        'websites': [],
        'director_names': []  # Add director names extraction
    }
    
    # Extract text content
    text_content = soup.get_text()
    
    # Extract phone numbers
    phones = PHONE_REGEX.findall(text_content)
    result['phone_numbers'] = list(set(phones))[:3]  # Limit to 3 most relevant
    
    # Extract emails
    emails = EMAIL_REGEX.findall(text_content)
    result['emails'] = list(set(emails))[:3]
    
    # Extract ratings
    ratings = RATING_REGEX.findall(text_content)
    if ratings:
        # Convert to float and take the first one
        try:
            rating = float(ratings[0].replace(',', '.'))
            result['ratings'] = [rating]
        except ValueError:
            pass
    
    # Extract review counts
    reviews = REVIEWS_REGEX.findall(text_content)
    if reviews:
        try:
            review_count = int(reviews[0])
            result['review_counts'] = [review_count]
        except ValueError:
            pass
    
    # Extract business names from specific Google Maps elements
    # Look for business name patterns in the HTML
    business_name_selectors = [
        '[data-value="title"]',
        '.x3AX1-LfntMc-header-title-title',
        '.DUwDvf.lfPIob',
        '.qrShPb',
        'h1'
    ]
    
    for selector in business_name_selectors:
        elements = soup.select(selector)
        for element in elements:
            text = element.get_text().strip()
            if text and len(text) > 2 and len(text) < 100:
                result['business_names'].append(text)
        if result['business_names']:
            break
    
    # Extract business type/category
    category_patterns = [
        r'(?:Restaurant|Café|Hôtel|Magasin|Bureau|Cabinet|Entreprise|Service|Agence)',
        r'(?:restaurant|café|hôtel|magasin|bureau|cabinet|entreprise|service|agence)'
    ]
    
    for pattern in category_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        result['business_types'].extend(matches[:2])
    
    # Look for website URLs
    url_pattern = re.compile(r'https?://[^\s<>"]+', re.IGNORECASE)
    urls = url_pattern.findall(text_content)
    # Filter out Google URLs and keep only business websites
    business_urls = [url for url in urls if not any(domain in url.lower() for domain in ['google.', 'maps.', 'youtube.', 'facebook.']) and '.' in url]
    result['websites'] = business_urls[:2]
    
    # Extract director/manager names
    # Look for common French business titles followed by names
    director_patterns = [
        r'(?:Directeur|Directrice|Gérant|Gérante|Président|Présidente|PDG|DG|Manager|Responsable)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(?:M\.|Mme|Monsieur|Madame)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'Contact\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'Propriétaire\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
    ]
    
    for pattern in director_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for match in matches:
            if len(match.split()) == 2:  # Ensure we have first name + last name
                result['director_names'].append(match.strip())
    
    # Remove duplicates and clean up
    for key in result:
        if isinstance(result[key], list):
            result[key] = list(dict.fromkeys(result[key]))  # Remove duplicates while preserving order
    
    return result

def _search_google_maps(query: str, session: httpx.Client) -> Dict[str, any]:
    """Search Google Maps for a business query."""
    result = {
        'query': query,
        'business_names': [],
        'phone_numbers': [],
        'emails': [],
        'ratings': [],
        'review_counts': [],
        'business_types': [],
        'websites': [],
        'director_names': [],  # Add director names
        'search_status': 'not_searched'
    }
    
    if not query.strip():
        result['search_status'] = 'empty_query'
        return result
    
    try:
        url = _build_google_maps_url(query)
        
        # Add random delay to avoid being blocked
        delay = random.uniform(*REQUEST_DELAY)
        time.sleep(delay)
        
        response = session.get(
            url,
            timeout=MAPS_TIMEOUT,
            follow_redirects=True
        )
        
        if response.status_code == 200:
            business_info = _extract_business_info_from_maps(response.text)
            result.update(business_info)
            result['search_status'] = 'success'
        else:
            result['search_status'] = f'http_error_{response.status_code}'
            
    except httpx.TimeoutException:
        result['search_status'] = 'timeout'
    except Exception as e:
        result['search_status'] = f'error_{type(e).__name__}'
    
    return result

def _merge_maps_results(search_results: List[Dict]) -> pd.DataFrame:
    """Merge Google Maps search results into a structured DataFrame."""
    if not search_results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(search_results)
    
    # Convert lists to strings for easier handling
    list_columns = ['business_names', 'phone_numbers', 'emails', 'business_types', 'websites', 'director_names']
    for col in list_columns:
        if col in df.columns:
            df[f'{col}_str'] = df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) and x else '')
    
    # Handle ratings and review counts (take first value if exists)
    if 'ratings' in df.columns:
        df['rating'] = df['ratings'].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    
    if 'review_counts' in df.columns:
        df['review_count'] = df['review_counts'].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    
    return df

def run(cfg: dict, ctx: dict) -> dict:
    """Run Google Maps search enrichment."""
    logger = ctx.get("logger")
    t0 = time.time()
    
    # Input/output paths
    input_path = Path(ctx["outdir"]) / "normalized.parquet"
    if not input_path.exists():
        # Fallback to CSV if parquet doesn't exist
        input_path = Path(ctx["outdir"]) / "normalized.csv"
        if not input_path.exists():
            # Look for normalized.csv in project root
            input_path = Path("/home/runner/work/projects/projects/normalized.csv")
            if not input_path.exists():
                # Check current working directory
                input_path = Path("normalized.csv")
                if not input_path.exists():
                    return {"status": "SKIPPED", "reason": "NO_NORMALIZED_DATA"}
    
    output_path = Path(ctx["outdir"]) / "google_maps_enriched.parquet"
    
    try:
        # Load input data
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        if df.empty:
            return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}
        
        # Check for required columns
        required_cols = ['raison_sociale', 'commune', 'code_postal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"status": "SKIPPED", "reason": f"MISSING_COLUMNS_{missing_cols}"}
        
        # Build search queries for each row
        df['maps_query'] = df.apply(_build_address_query, axis=1)
        
        # Filter to rows with valid queries
        valid_queries = df[df['maps_query'].str.strip() != '']
        
        if valid_queries.empty:
            return {"status": "SKIPPED", "reason": "NO_VALID_QUERIES"}
        
        # Get unique queries to avoid duplicate searches
        unique_queries = valid_queries['maps_query'].unique().tolist()
        
        if logger:
            logger.info(f"Starting Google Maps search for {len(unique_queries)} unique queries")
        
        # Perform searches with extensive rate limiting
        search_results = []
        
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        with httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=MAPS_TIMEOUT
        ) as session:
            
            # Process queries sequentially to avoid being blocked
            for i, query in enumerate(unique_queries):
                try:
                    if logger:
                        logger.debug(f"Searching Google Maps for query {i+1}/{len(unique_queries)}: {query[:50]}...")
                    
                    result = _search_google_maps(query, session)
                    search_results.append(result)
                    
                    # Additional delay between requests
                    if i < len(unique_queries) - 1:
                        delay = random.uniform(*REQUEST_DELAY)
                        time.sleep(delay)
                        
                except Exception as e:
                    if logger:
                        logger.error(f"Search failed for query {query}: {e}")
                    search_results.append({
                        'query': query,
                        'search_status': 'error'
                    })
        
        # Merge results
        search_df = _merge_maps_results(search_results)
        
        if search_df.empty:
            return {"status": "FAIL", "reason": "NO_SEARCH_RESULTS"}
        
        # Join with original data
        enriched_df = df.merge(
            search_df[['query', 'business_names_str', 'phone_numbers_str', 'emails_str', 
                      'business_types_str', 'websites_str', 'director_names_str', 'rating', 'review_count', 'search_status']], 
            left_on='maps_query', 
            right_on='query', 
            how='left'
        )
        
        # Clean up merge column
        enriched_df.drop(['query', 'maps_query'], axis=1, inplace=True)
        
        # Rename columns to have maps prefix for clarity
        column_mapping = {
            'business_names_str': 'maps_business_names',
            'phone_numbers_str': 'maps_phone_numbers', 
            'emails_str': 'maps_emails',
            'business_types_str': 'maps_business_types',
            'websites_str': 'maps_websites',
            'director_names_str': 'maps_director_names',
            'rating': 'maps_rating',
            'review_count': 'maps_review_count',
            'search_status': 'maps_search_status'
        }
        
        enriched_df.rename(columns=column_mapping, inplace=True)
        
        # Fill missing values
        string_cols = ['maps_business_names', 'maps_phone_numbers', 'maps_emails', 'maps_business_types', 'maps_websites', 'maps_director_names']
        for col in string_cols:
            if col in enriched_df.columns:
                enriched_df[col] = enriched_df[col].fillna('')
        
        if 'maps_search_status' in enriched_df.columns:
            enriched_df['maps_search_status'] = enriched_df['maps_search_status'].fillna('not_searched')
        
        # Save results
        table = pa.Table.from_pandas(enriched_df, preserve_index=False)
        with ParquetBatchWriter(output_path) as writer:
            writer.write_table(table)
        
        # Calculate statistics
        successful_searches = len([r for r in search_results if r.get('search_status') == 'success'])
        total_searches = len(search_results)
        
        elapsed = time.time() - t0
        
        if logger:
            logger.info(f"Google Maps enrichment completed in {elapsed:.1f}s")
            logger.info(f"Successfully enriched {successful_searches}/{total_searches} queries")
        
        return {
            "status": "OK",
            "records_processed": len(enriched_df),
            "searches_performed": total_searches,
            "successful_searches": successful_searches,
            "elapsed_time": elapsed,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Google Maps enrichment failed: {e}")
        return {"status": "FAIL", "error": str(e)}