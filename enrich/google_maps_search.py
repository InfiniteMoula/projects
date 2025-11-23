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

import os
import re
import time
import urllib.parse
import random
from pathlib import Path
from dataclasses import dataclass
import errno
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple
import pandas as pd
import pyarrow as pa
import httpx
from bs4 import BeautifulSoup

from proxy_manager import ProxyManager
from utils import budget_middleware
from utils.parquet import ParquetBatchWriter
from utils.state import SequentialRunState

# Configuration
GMAPS_WORKERS = int(os.getenv("GMAPS_WORKERS", "2"))  # concurrency
GMAPS_DELAY_MIN = float(os.getenv("GMAPS_DELAY_MIN", "0.5"))  # seconds
GMAPS_DELAY_MAX = float(os.getenv("GMAPS_DELAY_MAX", "1.0"))  # seconds
GMAPS_TIMEOUT = float(os.getenv("GMAPS_TIMEOUT", "8.0"))  # seconds

SKIP_IF_DOMAIN_AND_PHONE = os.getenv("GMAPS_SKIP_IF_DOMAIN_AND_PHONE", "1") == "1"

MAPS_TIMEOUT = GMAPS_TIMEOUT
REQUEST_DELAY = (GMAPS_DELAY_MIN, GMAPS_DELAY_MAX)
MAX_WORKERS = GMAPS_WORKERS  # kept for compatibility with existing code paths
RETRY_COUNT = 2

PROXY_MANAGER = ProxyManager()


def _proxy_settings_for_httpx() -> Optional[Dict[str, str]]:
    """
    Return httpx-compatible proxy settings from ProxyManager.

    Falls back to as_requests() for compatibility while waiting for
    ProxyManager.as_httpx() to be available everywhere.
    """

    if hasattr(PROXY_MANAGER, "as_httpx"):
        return PROXY_MANAGER.as_httpx()  # type: ignore[attr-defined]
    proxies = PROXY_MANAGER.as_requests()
    if not proxies:
        return None
    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https", http_proxy)
    proxy_mapping: Dict[str, str] = {}
    if http_proxy:
        proxy_mapping["http://"] = http_proxy
    if https_proxy:
        proxy_mapping["https://"] = https_proxy
    return proxy_mapping or None

# User agent that looks like a regular browser
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

# Regular expressions for extracting information
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
RATING_REGEX = re.compile(r'(\d+[.,]\d+)\s*(?:étoiles?|stars?|★)', re.IGNORECASE)
REVIEWS_REGEX = re.compile(r'(\d+)\s*(?:avis|reviews?|commentaires?)', re.IGNORECASE)

@dataclass
class GMapsStats:
    raw_count: int = 0
    after_cleanup: int = 0
    unique_candidates: int = 0
    queries_sent: int = 0
    queries_skipped_dup: int = 0
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.time()

    def elapsed_minutes(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return (time.time() - self.start_time) / 60.0

    def requests_per_minute(self) -> float:
        minutes = self.elapsed_minutes()
        if minutes <= 0:
            return 0.0
        return self.queries_sent / minutes

def _prepare_addresses_dataframe(df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, Optional[str]]:
    """Clean, filter, and deduplicate addresses before querying Google Maps."""
    df_raw = df
    raw_count = len(df_raw)
    df_clean = df_raw.copy()

    address_col = None
    for cand in ["adresse_complete", "address", "full_address", "maps_query", "adresse"]:
        if cand in df_clean.columns:
            address_col = cand
            break

    if address_col is None:
        if logger:
            logger.warning("Google Maps address column not found, skipping cleanup")
        return df_raw, None

    df_clean[address_col] = df_clean[address_col].astype("string").str.strip()
    df_clean = df_clean[df_clean[address_col].notna() & (df_clean[address_col].str.len() >= 10)]

    subset = [c for c in ["adresse_complete", "code_postal", "ville"] if c in df_clean.columns]
    if subset:
        df_clean = df_clean.drop_duplicates(subset=subset)

    if SKIP_IF_DOMAIN_AND_PHONE and {"domain_root", "telephone_norm"}.issubset(df_clean.columns):
        df_clean = df_clean[~(df_clean["domain_root"].notna() & df_clean["telephone_norm"].notna())]

    filtered_count = len(df_clean)
    if logger:
        logger.info(
            "Google Maps address prep | raw=%d after_cleanup=%d",
            raw_count,
            filtered_count,
        )

    return df_clean, address_col

def _safe_save_state(
    action: Callable[[], None],
    logger: Optional[logging.Logger],
    max_retries: int = 3,
    delay_sec: float = 0.5,
) -> None:
    """Persist state with retries on transient Windows access-denied errors."""
    log = logger or logging.getLogger(__name__)
    for attempt in range(1, max_retries + 1):
        try:
            action()
            return
        except PermissionError as e:
            winerror = getattr(e, "winerror", None)
            if winerror == 5 or e.errno == errno.EACCES:
                log.warning(
                    "Failed to save Google Maps state (attempt %d/%d): %r",
                    attempt,
                    max_retries,
                    e,
                )
                if attempt < max_retries:
                    time.sleep(delay_sec)
                    continue
                log.error(
                    "Giving up on saving Google Maps state after %d attempts: %r",
                    max_retries,
                    e,
                )
                return
            raise
        except OSError as e:
            winerror = getattr(e, "winerror", None)
            if winerror == 5 or e.errno == errno.EACCES:
                log.warning(
                    "Failed to save Google Maps state (attempt %d/%d, OSError): %r",
                    attempt,
                    max_retries,
                    e,
                )
                if attempt < max_retries:
                    time.sleep(delay_sec)
                    continue
                log.error(
                    "Giving up on saving Google Maps state after %d attempts (OSError): %r",
                    max_retries,
                    e,
                )
                return
            raise

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

def _search_google_maps(
    query: str,
    session: httpx.Client,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> Dict[str, any]:
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
        
        response = session.get(url, timeout=MAPS_TIMEOUT, follow_redirects=True)
        if request_tracker:
            request_tracker(len(response.content or b""))

        if response.status_code == 200:
            business_info = _extract_business_info_from_maps(response.text)
            result.update(business_info)
            result['search_status'] = 'success'
        else:
            result['search_status'] = f'http_error_{response.status_code}'
            
    except budget_middleware.BudgetExceededError:
        raise
    except httpx.TimeoutException:
        if request_tracker:
            request_tracker(0)
        result['search_status'] = 'timeout'
    except httpx.HTTPError as exc:
        if request_tracker:
            response_obj = getattr(exc, "response", None)
            size = len(response_obj.content or b"") if response_obj and response_obj.content else 0
            request_tracker(size)
        result['search_status'] = f'error_{exc.__class__.__name__}'
    except Exception as e:
        if isinstance(e, budget_middleware.BudgetExceededError):
            raise
        if request_tracker:
            request_tracker(0)
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
    if logger:
        logger.info(
            "Google Maps config | workers=%s delay=(%.2f, %.2f) timeout=%.1fs skip_if_domain_and_phone=%s",
            GMAPS_WORKERS,
            GMAPS_DELAY_MIN,
            GMAPS_DELAY_MAX,
            GMAPS_TIMEOUT,
            SKIP_IF_DOMAIN_AND_PHONE,
        )
        logger.info("Google Maps proxy enabled: %s", PROXY_MANAGER.enabled)
    t0 = time.time()

    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    # Input/output paths - first check for database.csv from address step
    database_path = outdir / "database.csv"
    input_path = outdir / "normalized.parquet"
    
    if not database_path.exists():
        # Fallback to old behavior if database.csv doesn't exist
        if not input_path.exists():
            input_path = outdir / "normalized.csv"
            if not input_path.exists():
                return {"status": "SKIPPED", "reason": "NO_DATABASE_OR_NORMALIZED_DATA"}
        database_df = None
    else:
        # Load database.csv with addresses
        database_df = pd.read_csv(database_path)
        if database_df.empty:
            return {"status": "SKIPPED", "reason": "EMPTY_DATABASE"}
    
    # Also need the main dataset for final merge
    if not input_path.exists():
        input_path = outdir / "normalized.csv"
        if not input_path.exists():
            return {"status": "SKIPPED", "reason": "NO_NORMALIZED_DATA"}

    request_tracker = ctx.get("request_tracker")
    output_path = outdir / "google_maps_enriched.parquet"
    stats = GMapsStats()
    raw_address_count = 0
    clean_address_count = 0
    
    try:
        # Load main input data
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        if df.empty:
            return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}
        
        addresses_df: Optional[pd.DataFrame] = None
        address_col: Optional[str] = None

        # Use database.csv if available, otherwise fall back to building addresses from components
        if database_df is not None:
            raw_address_count = len(database_df)
            addresses_df, address_col = _prepare_addresses_dataframe(database_df, logger)
            if address_col is None:
                addresses_df = database_df
            clean_address_count = len(addresses_df)
            if logger:
                logger.info("Using %d addresses from database.csv", len(addresses_df))
        else:
            # Fall back to old behavior
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
            
            raw_address_count = len(valid_queries)
            addresses_df, address_col = _prepare_addresses_dataframe(valid_queries, logger)
            if address_col is None:
                address_col = "maps_query"
                addresses_df = valid_queries
            clean_address_count = len(addresses_df)
            if logger:
                logger.info("Using %d addresses from normalized data", len(addresses_df))
        
        if addresses_df is None:
            return {"status": "SKIPPED", "reason": "NO_ADDRESS_DATA"}

        if address_col is None:
            if logger:
                logger.warning("No address column detected, cannot prepare queries")
            return {"status": "SKIPPED", "reason": "NO_ADDRESS_COLUMN"}

        seen_queries: Set[str] = set()
        unique_queries: List[str] = []
        for _, row in addresses_df.iterrows():
            raw_value = row.get(address_col, "")
            query = str(raw_value or "").strip()
            if not query:
                continue

            parts = []
            for col in ["adresse_complete", "code_postal", "ville"]:
                if col in addresses_df.columns and pd.notna(row.get(col)):
                    parts.append(str(row.get(col)))
            query_key = "|".join(parts) if parts else query

            if query_key in seen_queries:
                if logger:
                    logger.debug("Skipping duplicate Google Maps query: %s", query_key)
                stats.queries_skipped_dup += 1
                continue

            seen_queries.add(query_key)
            unique_queries.append(query)

        stats.raw_count = raw_address_count
        stats.after_cleanup = clean_address_count
        stats.unique_candidates = len(unique_queries)
        if stats.unique_candidates:
            stats.start()

        if not unique_queries:
            return {"status": "SKIPPED", "reason": "NO_QUERIES"}

        state = SequentialRunState(outdir / "google_maps_state.json")
        _safe_save_state(lambda: state.set_metadata(total=len(unique_queries)), logger)

        completed_map = state.metadata.get("completed_extra")
        if not isinstance(completed_map, dict):
            completed_map = {}

        pending_queries = set(state.pending(unique_queries))

        if logger:
            logger.info(
                "Starting Google Maps search for %d unique addresses (%d pending)",
                len(unique_queries),
                len(pending_queries),
            )

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        proxies = _proxy_settings_for_httpx()
        with httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=MAPS_TIMEOUT,
            proxies=proxies,
        ) as session:
            for idx, query in enumerate(unique_queries):
                if query not in pending_queries:
                    continue

                if logger:
                    logger.debug("Searching Google Maps for query %d/%d: %s", idx + 1, len(unique_queries), query[:50])

                _safe_save_state(lambda: state.mark_started(query), logger)
                try:
                    stats.queries_sent += 1
                    if logger and stats.queries_sent % 500 == 0:
                        logger.info(
                            "Google Maps progress | sent=%d skipped_dup=%d elapsed_min=%.2f req_per_min=%.2f",
                            stats.queries_sent,
                            stats.queries_skipped_dup,
                            stats.elapsed_minutes(),
                            stats.requests_per_minute(),
                        )
                    result = _search_google_maps(query, session, request_tracker=request_tracker)
                    _safe_save_state(lambda: state.mark_completed(query, extra=result), logger)
                    completed_map[query] = result
                except budget_middleware.BudgetExceededError:
                    _safe_save_state(lambda: state.mark_failed(query, "budget_exceeded"), logger)
                    _safe_save_state(lambda: state.set_metadata(last_error=query, processed=idx), logger)
                    raise
                except Exception as exc:
                    if logger:
                        logger.error("Search failed for query %s: %s", query, exc)
                    failure_result = {
                        "query": query,
                        "business_names": [],
                        "phone_numbers": [],
                        "emails": [],
                        "ratings": [],
                        "review_counts": [],
                        "business_types": [],
                        "websites": [],
                        "director_names": [],
                        "search_status": f"error_{type(exc).__name__}",
                    }
                    _safe_save_state(lambda: state.mark_completed(query, extra=failure_result), logger)
                    completed_map[query] = failure_result
                finally:
                    pending_queries.discard(query)
                    if idx < len(unique_queries) - 1 and query in completed_map:
                        delay = random.uniform(*REQUEST_DELAY)
                        time.sleep(delay)

        meta_completed = state.metadata.get("completed_extra")
        if isinstance(meta_completed, dict):
            completed_map = meta_completed
        search_results = [
            completed_map[query] for query in unique_queries if isinstance(completed_map.get(query), dict)
        ]

        # Merge results
        search_df = _merge_maps_results(search_results)
        
        if search_df.empty:
            return {"status": "FAIL", "reason": "NO_SEARCH_RESULTS"}
        
        # Join with original data - different logic depending on data source
        if database_df is not None:
            # When using database.csv, merge on address
            # First, create a mapping from address to enrichment data
            address_to_enrichment = {}
            for _, row in search_df.iterrows():
                address = row['query']
                address_to_enrichment[address] = {
                    'business_names_str': row.get('business_names_str', ''),
                    'phone_numbers_str': row.get('phone_numbers_str', ''),
                    'emails_str': row.get('emails_str', ''),
                    'business_types_str': row.get('business_types_str', ''),
                    'websites_str': row.get('websites_str', ''),
                    'director_names_str': row.get('director_names_str', ''),
                    'rating': row.get('rating', None),
                    'review_count': row.get('review_count', None),
                    'search_status': row.get('search_status', 'not_searched')
                }
            
            # Add enrichment data to main dataset based on constructed addresses
            enriched_df = df.copy()
            for idx, row in df.iterrows():
                # Build address for this row (same order as address parsing step)
                address_parts = []
                # Column AW: numero_voie
                if 'numero_voie' in row and pd.notna(row['numero_voie']):
                    address_parts.append(str(row['numero_voie']).strip())
                # Column BB: type_voie
                if 'type_voie' in row and pd.notna(row['type_voie']):
                    address_parts.append(str(row['type_voie']).strip())
                # Column AI: libelle_voie
                if 'libelle_voie' in row and pd.notna(row['libelle_voie']):
                    address_parts.append(str(row['libelle_voie']).strip())
                
                # Column BD: ville
                city = None
                if 'ville' in row and pd.notna(row['ville']):
                    city = str(row['ville']).strip()
                elif 'commune' in row and pd.notna(row['commune']):
                    city = str(row['commune']).strip()
                if city:
                    address_parts.append(city)
                # Column D: code_postal
                if 'code_postal' in row and pd.notna(row['code_postal']):
                    address_parts.append(str(row['code_postal']).strip())
                
                address = ' '.join(address_parts)
                
                # Get enrichment data for this address
                enrichment = address_to_enrichment.get(address, {
                    'business_names_str': '',
                    'phone_numbers_str': '',
                    'emails_str': '',
                    'business_types_str': '',
                    'websites_str': '',
                    'director_names_str': '',
                    'rating': None,
                    'review_count': None,
                    'search_status': 'not_searched'
                })
                
                # Add enrichment columns
                for key, value in enrichment.items():
                    enriched_df.at[idx, key] = value
                    
        else:
            # Original logic for when not using database.csv
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
        
        _safe_save_state(
            lambda: state.set_metadata(
                successful_searches=successful_searches,
                records=len(search_results),
                last_output=str(output_path),
            ),
            logger,
        )

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
    finally:
        if logger and (
            stats.raw_count
            or stats.after_cleanup
            or stats.unique_candidates
            or stats.queries_sent
            or stats.queries_skipped_dup
        ):
            logger.info(
                "Google Maps stats | raw=%d after_cleanup=%d unique_candidates=%d sent=%d skipped_dup=%d elapsed_min=%.2f req_per_min=%.2f",
                stats.raw_count,
                stats.after_cleanup,
                stats.unique_candidates,
                stats.queries_sent,
                stats.queries_skipped_dup,
                stats.elapsed_minutes(),
                stats.requests_per_minute(),
            )
