#!/usr/bin/env python3
"""
Address search enrichment module.

Searches Google.fr and Bing.com for business addresses to extract:
- Business names
- Phone numbers

Results are merged into the dataset to enrich contact information.
"""

import re
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import httpx
from bs4 import BeautifulSoup

from utils.parquet import ParquetBatchWriter, iter_batches

# Configuration
SEARCH_TIMEOUT = 10.0
REQUEST_DELAY = (1.0, 2.5)  # Random delay between requests (min, max) in seconds
MAX_WORKERS = 2  # Limited to avoid being blocked
RETRY_COUNT = 2

# User agent that looks like a regular browser
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Regular expressions for extracting contact information
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

def _build_search_urls(address: str) -> List[Tuple[str, str]]:
    """Build search URLs for Google.fr and Bing.com."""
    encoded_address = urllib.parse.quote_plus(address)
    
    urls = [
        ("google", f"https://www.google.fr/search?q={encoded_address}"),
        ("bing", f"https://www.bing.com/search?q={encoded_address}")
    ]
    
    return urls

def _extract_business_info_from_html(html_content: str, search_engine: str) -> Dict[str, List[str]]:
    """Extract business information from search result HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = {
        'business_names': [],
        'phone_numbers': [],
        'emails': []
    }
    
    # Extract text content
    text_content = soup.get_text()
    
    # Extract phone numbers (more permissive pattern)
    phones = PHONE_REGEX.findall(text_content)
    # Also extract simpler phone patterns
    simple_phone_pattern = re.compile(r'0[1-9](?:[.\-\s]?\d{2}){4}')
    simple_phones = simple_phone_pattern.findall(text_content)
    all_phones = phones + simple_phones
    result['phone_numbers'] = list(set(all_phones))
    
    # Extract emails
    emails = EMAIL_REGEX.findall(text_content)
    result['emails'] = list(set(emails))
    
    # Extract business names (this is search engine specific)
    if search_engine == "google":
        # Look for business names in Google results
        business_names = []
        
        # Google result titles (more general pattern)
        for title in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = title.get_text(strip=True)
            if text and len(text) < 200:  # Likely a business name
                business_names.append(text)
        
        # Look for any element that might contain business names
        for element in soup.find_all(['div', 'span', 'p']):
            text = element.get_text(strip=True)
            if text and 20 < len(text) < 100 and any(word in text.lower() for word in ['sa', 'sarl', 'cabinet', 'expert', 'comptable']):
                business_names.append(text)
                
        result['business_names'] = list(set(business_names[:5]))  # Limit to first 5
        
    elif search_engine == "bing":
        # Look for business names in Bing results
        business_names = []
        
        # Bing result titles
        for title in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = title.get_text(strip=True)
            if text and len(text) < 200:
                business_names.append(text)
        
        # Bing structured results
        for element in soup.find_all(['div', 'span', 'p']):
            text = element.get_text(strip=True)
            if text and 20 < len(text) < 100 and any(word in text.lower() for word in ['sa', 'sarl', 'cabinet', 'expert', 'comptable']):
                business_names.append(text)
                
        result['business_names'] = list(set(business_names[:5]))  # Limit to first 5
    
    return result

def _search_address(address: str, session: httpx.Client) -> Dict[str, any]:
    """Search for an address on Google.fr and Bing.com."""
    import random
    
    result = {
        'address': address,
        'google_business_names': [],
        'google_phones': [],
        'google_emails': [],
        'bing_business_names': [],
        'bing_phones': [],
        'bing_emails': [],
        'search_status': 'success'
    }
    
    urls = _build_search_urls(address)
    
    for search_engine, url in urls:
        try:
            # Add random delay to avoid being blocked
            delay = random.uniform(*REQUEST_DELAY)
            time.sleep(delay)
            
            response = session.get(url, timeout=SEARCH_TIMEOUT)
            response.raise_for_status()
            
            # Extract information from HTML
            info = _extract_business_info_from_html(response.text, search_engine)
            
            # Store results by search engine
            result[f'{search_engine}_business_names'] = info['business_names']
            result[f'{search_engine}_phones'] = info['phone_numbers'] 
            result[f'{search_engine}_emails'] = info['emails']
            
        except Exception as e:
            print(f"Error searching {search_engine} for '{address}': {e}")
            result['search_status'] = f'partial_error_{search_engine}'
    
    return result

def _merge_search_results(search_results: List[Dict]) -> pd.DataFrame:
    """Merge search results into a structured DataFrame."""
    if not search_results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(search_results)
    
    # Combine results from both search engines
    def combine_lists(row, field_prefix):
        google_list = row.get(f'google_{field_prefix}', [])
        bing_list = row.get(f'bing_{field_prefix}', [])
        combined = list(set(google_list + bing_list))
        return combined[:3] if combined else []  # Limit to top 3 results
    
    df['found_business_names'] = df.apply(lambda row: combine_lists(row, 'business_names'), axis=1)
    df['found_phones'] = df.apply(lambda row: combine_lists(row, 'phones'), axis=1)
    df['found_emails'] = df.apply(lambda row: combine_lists(row, 'emails'), axis=1)
    
    # Convert lists to strings for easier handling
    df['found_business_names_str'] = df['found_business_names'].apply(lambda x: '; '.join(x) if x else '')
    df['found_phones_str'] = df['found_phones'].apply(lambda x: '; '.join(x) if x else '')
    df['found_emails_str'] = df['found_emails'].apply(lambda x: '; '.join(x) if x else '')
    
    return df

def _build_address_from_components(row: pd.Series) -> str:
    """Build a complete address from individual components."""
    address_parts = []
    
    # Add street number (numero_voie)
    if 'numero_voie' in row and pd.notna(row['numero_voie']):
        address_parts.append(str(row['numero_voie']).strip())
    
    # Add street type (type_voie)  
    if 'type_voie' in row and pd.notna(row['type_voie']):
        address_parts.append(str(row['type_voie']).strip())
    
    # Add street name (libelle_voie)
    if 'libelle_voie' in row and pd.notna(row['libelle_voie']):
        address_parts.append(str(row['libelle_voie']).strip())
    
    # Add city (ville/commune)
    city = None
    if 'ville' in row and pd.notna(row['ville']):
        city = str(row['ville']).strip()
    elif 'commune' in row and pd.notna(row['commune']):
        city = str(row['commune']).strip()
    
    if city:
        address_parts.append(city)
    
    # Add postal code (code_postal)
    if 'code_postal' in row and pd.notna(row['code_postal']):
        address_parts.append(str(row['code_postal']).strip())
    
    return ' '.join(address_parts)

def run(cfg: dict, ctx: dict) -> dict:
    """Run address extraction and database creation step."""
    logger = ctx.get("logger")
    t0 = time.time()
    
    # Input/output paths
    input_path = Path(ctx["outdir"]) / "normalized.parquet"
    if not input_path.exists():
        # Fallback to CSV if parquet doesn't exist
        input_path = Path(ctx["outdir"]) / "normalized.csv"
        if not input_path.exists():
            return {"status": "SKIPPED", "reason": "NO_NORMALIZED_DATA"}
    
    database_path = Path(ctx["outdir"]) / "database.csv"
    
    try:
        # Load input data
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        if df.empty:
            return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}
        
        if logger:
            logger.info(f"Processing {len(df)} records for address extraction")
        
        # Build addresses from components (numero_voie + type_voie + libelle_voie + ville + code_postal)
        df['adresse_complete'] = df.apply(_build_address_from_components, axis=1)
        
        # Create database with addresses and company names in the same order
        database_df = pd.DataFrame({
            'index': df.index,
            'adresse': df['adresse_complete'],
            'company_name': df.get('denomination', df.get('raison_sociale', '')),
            'siren': df.get('siren', ''),
            'siret': df.get('siret', '')
        })
        
        # Filter out rows with empty addresses
        database_df = database_df[database_df['adresse'].str.strip() != '']
        
        if database_df.empty:
            return {"status": "SKIPPED", "reason": "NO_VALID_ADDRESSES"}
        
        # Save database.csv
        database_df.to_csv(database_path, index=False)
        
        if logger:
            logger.info(f"Created database.csv with {len(database_df)} address entries")
        
        # For backward compatibility, also create the address enriched output
        # This step now just prepares the addresses - enrichment happens in google_maps step
        addresses_to_search = database_df['adresse'].unique().tolist()
        
        # Create a simple output for this step - actual enrichment moved to google_maps step
        result_df = df.copy()
        result_df['database_created'] = True
        result_df['addresses_extracted'] = len(addresses_to_search)
        
        # Save the address extraction output 
        output_path = Path(ctx["outdir"]) / "address_extracted.parquet"
        result_df.to_parquet(output_path)
        
        duration = time.time() - t0
        
        return {
            "status": "OK",
            "step": "enrich.address", 
            "records_processed": len(df),
            "addresses_extracted": len(addresses_to_search),
            "database_records": len(database_df),
            "duration_s": duration,
            "files": {
                "database": str(database_path),
                "output": str(output_path)
            }
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error in address extraction: {str(e)}")
        return {"status": "ERROR", "error": str(e)}