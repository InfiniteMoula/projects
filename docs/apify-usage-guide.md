# Apify Scrapers - Usage Guide

This guide explains how to effectively use the Apify scrapers for business data enrichment, including practical examples, best practices, and optimization strategies.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Usage Patterns](#usage-patterns)
3. [Configuration Examples](#configuration-examples)
4. [Cost Management](#cost-management)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Integration Patterns](#integration-patterns)

## Quick Start

### 1. Basic Setup

**Prerequisites:**
- Apify account with API token
- Sufficient Apify credits (minimum 100 credits recommended)
- Input data with addresses (`database.csv` from step 7)

**Environment Setup:**
```bash
# 1. Add API token to .env file
echo "APIFY_API_TOKEN=your_token_here" >> .env

# 2. Verify dependencies
pip install apify-client>=2.1.0

# 3. Test connection
python -c "
import os
from apify_client import ApifyClient
client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
print('✓ Apify connection successful')
"
```

### 2. Minimal Configuration

Create a job file with basic Apify settings:

```yaml
# minimal_apify_job.yaml
niche: "apify_test"

filters:
  naf_include: ["6920Z"]  # Legal services
  regions: ["75"]         # Paris

profile: "standard"

apify:
  enabled: true
  max_addresses: 5        # Start small for testing
  
  google_places:
    enabled: true
    max_places_per_search: 3
  
  google_maps_contacts:
    enabled: false        # Disable for initial test
  
  linkedin_premium:
    enabled: false        # Disable for initial test
```

### 3. First Run

```bash
# Test with dry run first
python builder_cli.py run-profile \
  --job minimal_apify_job.yaml \
  --input data/sirene_sample.parquet \
  --out out/apify_test \
  --profile standard \
  --sample 10 \
  --dry-run

# Actual run
python builder_cli.py run-profile \
  --job minimal_apify_job.yaml \
  --input data/sirene_sample.parquet \
  --out out/apify_test \
  --profile standard \
  --sample 10
```

## Usage Patterns

### 1. Development and Testing Pattern

**Use Case**: Testing new scrapers or configurations with minimal cost

```yaml
apify:
  enabled: true
  max_addresses: 2
  save_raw_results: true    # Keep for debugging
  
  google_places:
    enabled: true
    max_places_per_search: 1
  
  google_maps_contacts:
    enabled: false
  
  linkedin_premium:
    enabled: false
```

**Cost**: ~2-5 credits per run

### 2. Production Data Enrichment Pattern

**Use Case**: Full-scale business intelligence gathering

```yaml
apify:
  enabled: true
  max_addresses: 100
  save_raw_results: false   # Save space
  
  google_places:
    enabled: true
    max_places_per_search: 10
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 80
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 50
    max_profiles_per_company: 3
```

**Cost**: ~800-1200 credits per run

### 3. Contact-Focused Pattern

**Use Case**: Prioritizing contact information over executive data

```yaml
apify:
  enabled: true
  max_addresses: 50
  
  google_places:
    enabled: true
    max_places_per_search: 5
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 100  # Higher limit
  
  linkedin_premium:
    enabled: false                # Disable expensive scraper
```

**Cost**: ~200-300 credits per run

### 4. Executive Research Pattern

**Use Case**: Focus on leadership and decision-maker information

```yaml
apify:
  enabled: true
  max_addresses: 20         # Fewer companies, more depth
  
  google_places:
    enabled: true
    max_places_per_search: 3
  
  google_maps_contacts:
    enabled: false
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 20
    max_profiles_per_company: 8  # More executives per company
```

**Cost**: ~600-800 credits per run

## Configuration Examples

### 1. Industry-Specific Configurations

#### Legal Services (NAF 6920Z)
```yaml
niche: "legal_services"

filters:
  naf_include: ["6920Z"]
  active_only: true
  regions: ["75", "92", "93", "94"]  # Greater Paris

apify:
  enabled: true
  max_addresses: 30
  
  google_places:
    enabled: true
    max_places_per_search: 5
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 25
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 15
    max_profiles_per_company: 2  # Focus on partners
```

#### Accounting Firms (NAF 6920Z variations)
```yaml
niche: "accounting_firms"

filters:
  naf_include: ["6920Z", "6911Z", "6912Z"]
  active_only: true

apify:
  enabled: true
  max_addresses: 50
  
  google_places:
    enabled: true
    max_places_per_search: 8
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 40
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 25
    max_profiles_per_company: 4  # Include senior accountants
```

#### Consulting Services
```yaml
niche: "consulting"

filters:
  naf_include: ["7022Z", "7021Z"]  # Management consulting
  
apify:
  enabled: true
  max_addresses: 40
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 35
    max_profiles_per_company: 6  # Consultants are key contacts
```

### 2. Regional Configurations

#### Paris Metropolitan Area
```yaml
filters:
  regions: ["75", "92", "93", "94", "95", "77", "78", "91"]

apify:
  max_addresses: 100  # High density area
  
  google_places:
    max_places_per_search: 15  # More competition
```

#### Rural Areas  
```yaml
filters:
  regions: ["03", "15", "23"]  # Rural departments

apify:
  max_addresses: 200  # Lower density, process more
  
  google_places:
    max_places_per_search: 5   # Fewer results expected
```

### 3. Budget-Based Configurations

#### Low Budget (< 100 credits)
```yaml
apify:
  enabled: true
  max_addresses: 10
  
  google_places:
    enabled: true
    max_places_per_search: 2
  
  google_maps_contacts:
    enabled: false
    
  linkedin_premium:
    enabled: false
```

#### Medium Budget (100-500 credits)
```yaml
apify:
  enabled: true
  max_addresses: 30
  
  google_places:
    enabled: true
    max_places_per_search: 5
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 20
    
  linkedin_premium:
    enabled: false  # Save for high-value only
```

#### High Budget (500+ credits)
```yaml
apify:
  enabled: true
  max_addresses: 50
  
  google_places:
    enabled: true
    max_places_per_search: 10
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 50
    
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 30
    max_profiles_per_company: 5
```

## Cost Management

### 1. Credit Usage Estimation

Use this calculator to estimate costs before running:

```python
def estimate_apify_costs(config):
    """Estimate Apify credit usage based on configuration."""
    
    max_addresses = config.get("max_addresses", 10)
    
    costs = {
        "google_places": 0,
        "google_maps_contacts": 0, 
        "linkedin_premium": 0
    }
    
    # Google Places: 1-5 credits per search
    if config.get("google_places", {}).get("enabled", False):
        avg_cost_per_search = 3
        costs["google_places"] = max_addresses * avg_cost_per_search
    
    # Contact Details: 1-3 credits per enrichment
    if config.get("google_maps_contacts", {}).get("enabled", False):
        max_enrichments = config.get("google_maps_contacts", {}).get("max_contact_enrichments", 20)
        avg_cost_per_enrichment = 2
        costs["google_maps_contacts"] = min(max_enrichments, max_addresses * 5) * avg_cost_per_enrichment
    
    # LinkedIn: 10-50 credits per search
    if config.get("linkedin_premium", {}).get("enabled", False):
        max_searches = config.get("linkedin_premium", {}).get("max_linkedin_searches", 10)
        avg_cost_per_search = 30
        costs["linkedin_premium"] = min(max_searches, max_addresses) * avg_cost_per_search
    
    total = sum(costs.values())
    
    return {
        "breakdown": costs,
        "total_estimated": total,
        "recommendation": _get_cost_recommendation(total)
    }

def _get_cost_recommendation(total_cost):
    """Get recommendation based on estimated cost."""
    if total_cost < 50:
        return "Low cost - safe for testing"
    elif total_cost < 200:
        return "Medium cost - good for development"
    elif total_cost < 500:
        return "High cost - production use"
    else:
        return "Very high cost - consider reducing limits"
```

### 2. Cost Control Strategies

#### Progressive Enrichment
Start with basic scrapers and add more expensive ones based on results:

```yaml
# Stage 1: Basic discovery (run first)
apify:
  google_places:
    enabled: true
  google_maps_contacts:
    enabled: false
  linkedin_premium:
    enabled: false

# Stage 2: Contact enrichment (if Stage 1 successful)  
apify:
  google_places:
    enabled: false  # Already have data
  google_maps_contacts:
    enabled: true
  linkedin_premium:
    enabled: false

# Stage 3: Executive research (for high-value targets)
apify:
  google_places:
    enabled: false
  google_maps_contacts:
    enabled: false  
  linkedin_premium:
    enabled: true
```

#### Quality-Based Filtering
Filter input data to focus on high-quality targets:

```python
def filter_high_value_targets(df):
    """Filter for high-value enrichment targets."""
    
    # Focus on larger companies (more employees)
    df = df[df.get('effectif_salarie', 0) >= 5]
    
    # Recent activity (updated in last 2 years)
    df = df[df['date_maj'].dt.year >= 2022]
    
    # Specific high-value industries
    high_value_naf = ["6920Z", "7022Z", "6201Z", "6202A"]
    df = df[df['naf_code'].isin(high_value_naf)]
    
    return df
```

### 3. Budget Monitoring

```python
def monitor_credit_usage():
    """Monitor Apify credit usage and send alerts."""
    
    from apify_client import ApifyClient
    
    client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    user_info = client.user().get()
    
    current_credits = user_info.get('monthlyUsage', {}).get('credits', 0)
    credit_limit = user_info.get('plan', {}).get('monthlyCredits', 1000)
    
    usage_percentage = (current_credits / credit_limit) * 100
    
    if usage_percentage > 80:
        print(f"⚠️  Warning: {usage_percentage:.1f}% of credits used")
    elif usage_percentage > 90:
        print(f"🚨 Critical: {usage_percentage:.1f}% of credits used")
    
    return {
        "current_usage": current_credits,
        "limit": credit_limit,
        "percentage": usage_percentage
    }
```

## Best Practices

### 1. Data Quality Optimization

#### Input Data Preparation
```python
def prepare_addresses_for_apify(df):
    """Optimize addresses for better Apify results."""
    
    # Clean and standardize addresses
    df['adresse_clean'] = df['adresse'].str.strip()
    df['adresse_clean'] = df['adresse_clean'].str.replace(r'\s+', ' ', regex=True)
    
    # Add postal codes if missing
    df['adresse_clean'] = df.apply(lambda row: 
        f"{row['adresse_clean']}, {row['code_postal']} {row['commune']}" 
        if row['code_postal'] not in row['adresse_clean'] 
        else row['adresse_clean'], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['adresse_clean'])
    
    return df
```

#### Company Name Optimization
```python
def prepare_company_names_for_linkedin(df):
    """Optimize company names for LinkedIn searches."""
    
    # Use best available company name
    df['linkedin_name'] = df['denomination'].fillna(
        df['raison_sociale'].fillna(
            df['company_name']
        )
    )
    
    # Clean company names
    df['linkedin_name'] = df['linkedin_name'].str.replace(r'\bSAS\b|\bSARL\b|\bSA\b', '', regex=True)
    df['linkedin_name'] = df['linkedin_name'].str.strip()
    
    return df
```

### 2. Error Recovery

#### Retry Logic
```python
def retry_failed_scrapers(original_config, failed_addresses):
    """Retry failed addresses with adjusted configuration."""
    
    retry_config = original_config.copy()
    
    # Reduce limits for retry
    retry_config["max_addresses"] = len(failed_addresses)
    retry_config["google_places"]["max_places_per_search"] = 3
    
    # Disable expensive scrapers for retry
    retry_config["linkedin_premium"]["enabled"] = False
    
    return retry_config
```

#### Partial Results Handling
```python
def handle_partial_results(results, expected_count):
    """Handle cases where some scrapers fail."""
    
    success_rate = len(results) / expected_count
    
    if success_rate >= 0.8:
        return "SUCCESS", "High success rate"
    elif success_rate >= 0.5:
        return "PARTIAL", "Acceptable success rate"
    else:
        return "FAILED", "Low success rate - check configuration"
```

### 3. Performance Optimization

#### Batch Processing
```python
def process_large_datasets_in_batches(df, batch_size=20):
    """Process large datasets efficiently."""
    
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i, batch_start in enumerate(range(0, len(df), batch_size)):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"Processing batch {i+1}/{total_batches} ({len(batch_df)} records)")
        
        # Process batch
        yield batch_df
        
        # Rate limiting between batches
        if i < total_batches - 1:  # Don't sleep after last batch
            time.sleep(2)
```

#### Caching Results
```python
def cache_apify_results(results, cache_key):
    """Cache results to avoid repeated API calls."""
    
    import json
    from pathlib import Path
    
    cache_dir = Path("cache/apify")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{cache_key}.json"
    
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return cache_file

def load_cached_results(cache_key):
    """Load cached results if available."""
    
    cache_file = Path(f"cache/apify/{cache_key}.json")
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    return None
```

## Troubleshooting

### 1. Common Issues and Solutions

#### Issue: "No input data found"
```bash
# Check for required files
ls -la out/*/database.csv
ls -la out/*/normalized.parquet

# Verify step 7 (address extraction) ran successfully
python builder_cli.py run-step --step enrich.address --job your_job.yaml
```

#### Issue: "APIFY_API_TOKEN environment variable is required"
```bash
# Check if token is set
echo $APIFY_API_TOKEN

# Set token permanently
echo "APIFY_API_TOKEN=your_token" >> ~/.bashrc
source ~/.bashrc

# Set for current session only
export APIFY_API_TOKEN=your_token
```

#### Issue: "Insufficient credits"
```python
# Check credit balance
from apify_client import ApifyClient
client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
user_info = client.user().get()
print(f"Credits: {user_info.get('monthlyUsage', {}).get('credits', 0)}")
```

#### Issue: "Actor timeout" errors
```yaml
# Increase timeouts in configuration
apify:
  google_places:
    timeout_seconds: 600  # 10 minutes
  linkedin_premium:
    timeout_seconds: 900  # 15 minutes
```

### 2. Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Run with debug output
python builder_cli.py run-profile \
  --job your_job.yaml \
  --input data/input.parquet \
  --out out/debug_run \
  --profile standard \
  --debug

# Check debug files
ls -la out/debug_run/apify_raw_results.json
ls -la out/debug_run/apify_debug.log
```

### 3. Data Validation

```python
def validate_apify_output(results_file):
    """Validate Apify output data quality."""
    
    import pandas as pd
    
    df = pd.read_parquet(results_file)
    
    validation_report = {
        "total_records": len(df),
        "records_with_places": (df['apify_places_found'] > 0).sum(),
        "records_with_phones": (df['apify_phones'] != '').sum(),
        "records_with_emails": (df['apify_emails'] != '').sum(),
        "records_with_linkedin": (df['apify_linkedin_profiles'] != '').sum(),
    }
    
    validation_report["success_rate"] = (
        validation_report["records_with_places"] / validation_report["total_records"]
    ) * 100
    
    return validation_report
```

## Integration Patterns

### 1. Pipeline Integration

#### Pre-Processing Hook
```python
def pre_apify_processing(df, config):
    """Prepare data before Apify processing."""
    
    # Filter for high-quality addresses
    df = df[df['adresse'].str.len() > 10]
    
    # Prioritize by business size
    df = df.sort_values('effectif_salarie', ascending=False)
    
    # Limit to budget
    max_addresses = config.get("apify", {}).get("max_addresses", 10)
    df = df.head(max_addresses)
    
    return df
```

#### Post-Processing Hook
```python
def post_apify_processing(df, config):
    """Process data after Apify enrichment."""
    
    # Calculate enrichment score
    df['enrichment_score'] = (
        (df['apify_phones'] != '').astype(int) * 3 +
        (df['apify_emails'] != '').astype(int) * 3 +
        (df['apify_websites'] != '').astype(int) * 2 +
        (df['apify_linkedin_profiles'] != '').astype(int) * 4
    )
    
    # Flag high-value contacts
    df['high_value'] = (
        (df['enrichment_score'] >= 8) &
        (df['apify_linkedin_profiles'] != '')
    )
    
    return df
```

### 2. Workflow Integration

#### Multi-Stage Processing
```yaml
# Stage 1: Discovery
stage1:
  apify:
    google_places:
      enabled: true
    google_maps_contacts:
      enabled: false
    linkedin_premium:
      enabled: false

# Stage 2: Contact Enrichment  
stage2:
  apify:
    google_places:
      enabled: false
    google_maps_contacts:
      enabled: true
    linkedin_premium:
      enabled: false

# Stage 3: Executive Research
stage3:
  apify:
    google_places:
      enabled: false
    google_maps_contacts:
      enabled: false
    linkedin_premium:
      enabled: true
```

#### Conditional Processing
```python
def conditional_apify_processing(df, budget_limit):
    """Adjust processing based on available budget."""
    
    if budget_limit < 100:
        # Basic processing only
        config = {
            "google_places": {"enabled": True},
            "google_maps_contacts": {"enabled": False},
            "linkedin_premium": {"enabled": False}
        }
    elif budget_limit < 500:
        # Contact enrichment
        config = {
            "google_places": {"enabled": True},
            "google_maps_contacts": {"enabled": True},
            "linkedin_premium": {"enabled": False}
        }
    else:
        # Full enrichment
        config = {
            "google_places": {"enabled": True},
            "google_maps_contacts": {"enabled": True},
            "linkedin_premium": {"enabled": True}
        }
    
    return config
```

This comprehensive usage guide provides practical patterns for implementing Apify scrapers effectively across different use cases and budgets.