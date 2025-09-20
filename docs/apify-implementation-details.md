# Apify Scrapers - Implementation Details

This document provides comprehensive technical details about the Apify scrapers implementation in the data enrichment pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Implementation Details](#implementation-details)
3. [Scraper Specifications](#scraper-specifications)
4. [Data Flow and Processing](#data-flow-and-processing)
5. [Error Handling and Reliability](#error-handling-and-reliability)
6. [Performance Considerations](#performance-considerations)
7. [Configuration Management](#configuration-management)

## Architecture Overview

The Apify integration (`api.apify_agents.py`) serves as a centralized orchestrator for three specialized web scrapers that extract business intelligence data from different sources.

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Data    │    │  Apify Client   │    │  Output Data    │
│                 │    │                 │    │                 │
│ • Addresses     │───▶│ • Google Places │───▶│ • Enriched CSV  │
│ • Company Names │    │ • Maps Contacts │    │ • Raw JSON      │
│ • SIRENE Data   │    │ • LinkedIn Pro  │    │ • Parquet Files │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Breakdown

1. **Input Layer**: Reads from `database.csv` (step 7 output) or normalized data
2. **Processing Layer**: Three parallel Apify scrapers with different specializations
3. **Output Layer**: Enriched data with contact details, ratings, and executive information

## Implementation Details

### Main Entry Point: `run(cfg, ctx)`

The main function coordinates the entire Apify enrichment process:

```python
def run(cfg: dict, ctx: dict) -> dict:
    # 1. Configuration validation
    # 2. Input data loading
    # 3. Apify client initialization  
    # 4. Parallel scraper execution
    # 5. Data merging and output
```

**Key Implementation Features:**

- **Graceful degradation**: If one scraper fails, others continue
- **Cost control**: Built-in limits for addresses and API calls
- **Parallel execution**: Scrapers run independently
- **Data validation**: Input sanitization and output validation

### Configuration Validation

```python
def _validate_config(cfg: dict) -> dict:
    """Validate and normalize Apify configuration."""
    apify_config = cfg.get("apify", {})
    
    # Default settings with cost controls
    defaults = {
        "enabled": True,
        "max_addresses": 10,
        "save_raw_results": False,
        "google_places": {"enabled": True, "max_places_per_search": 10},
        "google_maps_contacts": {"enabled": True, "max_contact_enrichments": 50},
        "linkedin_premium": {"enabled": True, "max_linkedin_searches": 20, "max_profiles_per_company": 5}
    }
    
    return {**defaults, **apify_config}
```

### Input Data Processing

The system handles multiple input sources with fallback logic:

1. **Primary**: `database.csv` from step 7 (address extraction)
2. **Fallback**: `normalized.parquet` from normalization step
3. **Validation**: Ensures required columns exist (`adresse`, company names)

```python
def _load_input_data(outdir: Path) -> pd.DataFrame:
    """Load and validate input data with fallback logic."""
    
    # Try database.csv first (step 7 output)
    database_path = outdir / "database.csv"
    if database_path.exists():
        return pd.read_csv(database_path)
    
    # Fallback to normalized data
    normalized_path = outdir / "normalized.parquet"
    if normalized_path.exists():
        return pd.read_parquet(normalized_path)
    
    raise FileNotFoundError("No input data found")
```

## Scraper Specifications

### 1. Google Places Crawler (`compass/crawler-google-places`)

**Purpose**: Primary business discovery and basic information extraction

**Input Format**:
```python
run_input = {
    "searchTerms": ["123 Rue de la Paix, 75001 Paris", ...],
    "language": "fr",
    "maxCrawledPlaces": 10,
    "exportPlaceUrls": True,
    "includeImages": False,
    "includeReviews": False,
    "maxReviews": 0
}
```

**Output Schema**:
```python
{
    "title": "Business Name",
    "address": "Full Address", 
    "url": "Google Maps URL",
    "phone": "+33 1 23 45 67 89",
    "website": "https://example.com",
    "category": "Business Category",
    "rating": 4.2,
    "reviewsCount": 156,
    "searchString": "Original search term"
}
```

**Implementation Notes**:
- Cost: ~1-5 credits per search
- Rate limiting: Built into Apify platform
- Timeout: Default 5 minutes per search
- Language: French (`fr`) for better local results

### 2. Google Maps Contact Details (`lukaskrivka/google-maps-with-contact-details`)

**Purpose**: Deep contact information extraction and business hours

**Input Format**:
```python
run_input = {
    "searchTerms": ["Business Name Address", "Google Maps URL", ...],
    "language": "fr", 
    "maxCrawledPlaces": 1,  # Already have the places
    "exportPlaceUrls": False
}
```

**Output Schema**:
```python
{
    "title": "Business Name",
    "phone": "+33 1 23 45 67 89",
    "email": "contact@business.com",
    "website": "https://business.com",
    "hours": "Mon-Fri 9AM-6PM",
    "address": "Full Address",
    "additionalInfo": "Extra contact details"
}
```

**Implementation Notes**:
- Cost: ~1-3 credits per enrichment
- Depends on Google Places results
- Extracts emails from business websites
- Provides detailed business hours

### 3. LinkedIn Premium Actor (`bebity/linkedin-premium-actor`)

**Purpose**: Executive and leadership information extraction

**Input Format**:
```python
run_input = {
    "searchTerms": ["Company Name", ...],
    "language": "fr",
    "maxProfiles": 5,
    "filters": {
        "positions": ["CEO", "CFO", "Director", "Founder", "Gérant", "Directeur", "Président"]
    }
}
```

**Output Schema**:
```python
{
    "fullName": "Jean Dupont",
    "position": "CEO",
    "companyName": "Business Name",
    "linkedinUrl": "https://linkedin.com/in/profile",
    "location": "Paris, France",
    "experience": "10+ years",
    "education": "University info"
}
```

**Implementation Notes**:
- Cost: ~10-50 credits per search (most expensive)
- Requires LinkedIn Premium subscription through Apify
- French executive titles support
- Limited to specified positions for cost control

## Data Flow and Processing

### 1. Data Ingestion

```python
def _prepare_input_data(df: pd.DataFrame, cfg: dict) -> tuple:
    """Prepare addresses and company names for processing."""
    
    # Extract and clean addresses
    addresses = df['adresse'].dropna().str.strip().tolist()
    
    # Extract company names with fallbacks
    company_cols = ['company_name', 'denomination', 'raison_sociale']
    company_names = []
    
    for col in company_cols:
        if col in df.columns:
            company_names = df[col].fillna('').str.strip().tolist()
            break
    
    # Apply cost controls
    max_addresses = cfg.get("max_addresses", 10)
    addresses = addresses[:max_addresses]
    company_names = company_names[:max_addresses]
    
    return addresses, company_names
```

### 2. Parallel Scraper Execution

The scrapers run in sequence but handle errors independently:

```python
def _execute_scrapers(addresses, companies, client, cfg):
    """Execute all enabled scrapers with error handling."""
    
    results = {
        'places': [],
        'contacts': [], 
        'linkedin': []
    }
    
    # Google Places (foundation for other scrapers)
    if cfg.get("google_places", {}).get("enabled", True):
        try:
            results['places'] = _run_google_places_crawler(addresses, client, cfg.get("google_places", {}))
        except Exception as e:
            logger.error(f"Google Places failed: {e}")
    
    # Contact Details (depends on places)
    if cfg.get("google_maps_contacts", {}).get("enabled", True) and results['places']:
        try:
            results['contacts'] = _run_google_maps_contact_details(results['places'], client, cfg.get("google_maps_contacts", {}))
        except Exception as e:
            logger.error(f"Contact details failed: {e}")
    
    # LinkedIn (independent)
    if cfg.get("linkedin_premium", {}).get("enabled", True) and companies:
        try:
            results['linkedin'] = _run_linkedin_premium_actor(companies, client, cfg.get("linkedin_premium", {}))
        except Exception as e:
            logger.error(f"LinkedIn scraping failed: {e}")
    
    return results
```

### 3. Data Merging and Enrichment

```python
def _merge_apify_results(addresses_df, places_results, contact_results, linkedin_results):
    """Merge all scraper results back into the original dataset."""
    
    enriched_df = addresses_df.copy()
    
    # Add enrichment columns
    new_columns = [
        'apify_places_found', 'apify_business_names', 'apify_phones',
        'apify_emails', 'apify_websites', 'apify_ratings', 
        'apify_categories', 'apify_executives', 'apify_linkedin_profiles'
    ]
    
    for col in new_columns:
        enriched_df[col] = ''
    
    # Process each address row
    for idx, row in enriched_df.iterrows():
        address = row['adresse']
        company_name = row.get('company_name', '')
        
        # Merge places data
        matching_places = [p for p in places_results if p.get('searchString') == address]
        if matching_places:
            enriched_df.at[idx, 'apify_places_found'] = len(matching_places)
            enriched_df.at[idx, 'apify_business_names'] = '; '.join([p.get('title', '') for p in matching_places])
            # ... additional field merging
        
        # Merge contact data
        # ... contact merging logic
        
        # Merge LinkedIn data  
        # ... LinkedIn merging logic
    
    return enriched_df
```

## Error Handling and Reliability

### 1. API Error Handling

```python
def _safe_api_call(client, actor_id, run_input, timeout=300):
    """Safe API call with timeout and retry logic."""
    
    try:
        run = client.actor(actor_id).call(run_input=run_input, timeout_secs=timeout)
        
        if run["status"] != "SUCCEEDED":
            raise Exception(f"Actor run failed with status: {run['status']}")
            
        return run
        
    except Exception as e:
        logger.error(f"API call failed for {actor_id}: {e}")
        return None
```

### 2. Data Validation

```python
def _validate_scraper_output(results, scraper_name):
    """Validate scraper output and clean invalid entries."""
    
    valid_results = []
    
    for item in results:
        if _is_valid_result(item, scraper_name):
            valid_results.append(_clean_result(item))
        else:
            logger.warning(f"Invalid result from {scraper_name}: {item}")
    
    return valid_results

def _is_valid_result(item, scraper_name):
    """Check if result contains minimum required fields."""
    
    required_fields = {
        'google_places': ['title', 'searchString'],
        'google_maps_contacts': ['title'],
        'linkedin_premium': ['fullName', 'companyName']
    }
    
    return all(field in item for field in required_fields.get(scraper_name, []))
```

### 3. Graceful Degradation

The system continues processing even if individual scrapers fail:

- If Google Places fails: Contact enrichment skipped, LinkedIn continues
- If Contact Details fails: Basic place info retained, LinkedIn continues  
- If LinkedIn fails: Business info still enriched from Google scrapers
- If all fail: Original data returned with error status

## Performance Considerations

### 1. Cost Optimization

**Built-in Limits**:
```python
default_limits = {
    "max_addresses": 10,           # Total addresses processed
    "max_places_per_search": 5,    # Results per address search
    "max_contact_enrichments": 20, # Contact detail calls
    "max_linkedin_searches": 5,    # LinkedIn company searches  
    "max_profiles_per_company": 3  # Executives per company
}
```

**Credit Usage Estimation**:
- Google Places: 5 addresses × 5 credits = 25 credits
- Contact Details: 20 places × 2 credits = 40 credits  
- LinkedIn: 5 companies × 30 credits = 150 credits
- **Total: ~215 credits per run**

### 2. Performance Optimization

**Batch Processing**:
```python
def _batch_process(items, batch_size=10):
    """Process items in batches to avoid API rate limits."""
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        time.sleep(1)  # Rate limiting
```

**Parallel Execution** (Future Enhancement):
```python
# Future implementation for parallel scraper execution
async def _run_scrapers_parallel(addresses, companies, client, cfg):
    """Run scrapers in parallel for better performance."""
    
    tasks = []
    
    if cfg.get("google_places", {}).get("enabled"):
        tasks.append(_run_google_places_async(addresses, client, cfg))
    
    if cfg.get("linkedin_premium", {}).get("enabled") and companies:
        tasks.append(_run_linkedin_premium_async(companies, client, cfg))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 3. Memory Management

**Stream Processing for Large Datasets**:
```python
def _process_large_dataset(df, chunk_size=1000):
    """Process large datasets in chunks to manage memory."""
    
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]
        
        yield chunk
```

## Configuration Management

### 1. Environment Variables

```bash
# Required
APIFY_API_TOKEN=your_apify_token_here

# Optional performance tuning
APIFY_REQUEST_TIMEOUT=300
APIFY_RETRY_COUNT=3
APIFY_RATE_LIMIT_DELAY=1
```

### 2. YAML Configuration Schema

```yaml
apify:
  enabled: true
  max_addresses: 50
  save_raw_results: true
  
  # Cost control per scraper
  google_places:
    enabled: true
    max_places_per_search: 10
    timeout_seconds: 300
    
  google_maps_contacts:
    enabled: true  
    max_contact_enrichments: 50
    timeout_seconds: 300
    
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 20
    max_profiles_per_company: 5
    timeout_seconds: 600  # LinkedIn is slower
    
  # Advanced settings
  retry_settings:
    max_retries: 3
    retry_delay: 5
    exponential_backoff: true
    
  output_settings:
    save_individual_results: false
    compress_raw_json: true
    include_debug_info: false
```

### 3. Dynamic Configuration

```python
def _build_runtime_config(base_cfg, ctx):
    """Build runtime configuration based on context."""
    
    config = base_cfg.copy()
    
    # Adjust limits based on available credits
    if ctx.get("budget_mode") == "minimal":
        config["max_addresses"] = min(config["max_addresses"], 5)
        config["linkedin_premium"]["enabled"] = False
    
    # Adjust for dry run mode
    if ctx.get("dry_run"):
        for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
            config[scraper]["enabled"] = False
    
    return config
```

This implementation provides a robust, scalable foundation for business data enrichment using Apify's powerful scraping infrastructure.