# Apify Platform Setup & Usage Guide

Complete guide for integrating and using Apify platform scrapers for business intelligence gathering through Google Places, Google Maps, and LinkedIn.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Configuration](#setup-and-configuration)
3. [Getting Started](#getting-started)
4. [Usage Patterns](#usage-patterns)
5. [Configuration Examples](#configuration-examples)
6. [Cost Management](#cost-management)
7. [Best Practices](#best-practices)
8. [Integration Patterns](#integration-patterns)

## Overview

The Apify integration provides professional-grade scrapers for:
- **Google Places Crawler**: Business discovery and basic information
- **Google Maps Contact Details**: Enhanced contact information (phones, emails, websites)
- **LinkedIn Premium Actor**: Executive information (CEO, CFO, Directors)

### Key Benefits

- **Professional Quality**: Enterprise-grade scrapers with high reliability
- **Cost Effective**: Pay-per-use model with precise cost control
- **Comprehensive Data**: Multiple data sources in a single integration
- **Quality Assurance**: Built-in validation and confidence scoring

## Setup and Configuration

### 1. Get Apify API Token

1. Create account at [Apify Console](https://console.apify.com)
2. Navigate to [Account Integrations](https://console.apify.com/account/integrations)
3. Copy your API token

### 2. Environment Configuration

Add your API token to the environment file:

```bash
# In .env file
APIFY_API_TOKEN=your_apify_token_here
```

### 3. Verify Setup

Test your API connection:

```bash
python -c "from apify_client import ApifyClient; import os; print(ApifyClient(os.getenv('APIFY_API_TOKEN')).user().get())"
```

## Getting Started

### Basic Job Configuration

Add Apify configuration to your job YAML file:

```yaml
# Apify scrapers configuration
apify:
  enabled: true  # Enable/disable Apify agents
  max_addresses: 50  # Limit addresses processed (cost control)
  save_raw_results: true  # Save raw JSON results for debugging
  
  # Google Places Crawler (compass/crawler-google-places)
  google_places:
    enabled: true
    max_places_per_search: 10
  
  # Google Maps with Contact Details (lukaskrivka/google-maps-with-contact-details)
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 50
  
  # LinkedIn Premium Actor (bebity/linkedin-premium-actor)
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 20
    max_profiles_per_company: 5
```

### First Run

```bash
# Test with minimal configuration
python builder_cli.py run-profile \
  --job jobs/apify_test.yaml \
  --input data/sample.parquet \
  --out out/apify_test \
  --profile standard \
  --sample 10 \
  --debug
```

### Pipeline Integration

Apify integration runs automatically as part of the standard and deep profiles:

1. **Step 7**: Address extraction creates `database.csv`
2. **Step 2b**: Apify agents process addresses from `database.csv`
3. **Subsequent steps**: Use Apify enriched data for further processing

## Usage Patterns

### 1. Basic Business Enrichment

```yaml
apify:
  enabled: true
  max_addresses: 100
  google_places:
    enabled: true
    max_places_per_search: 10
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 100
  linkedin_premium:
    enabled: false  # Disable to reduce costs
```

**Use case**: Contact discovery and basic business information
**Cost**: ~50-150 credits for 100 businesses

### 2. Executive Intelligence Gathering

```yaml
apify:
  enabled: true
  max_addresses: 50
  google_places:
    enabled: true
    max_places_per_search: 5
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 50
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 30
    max_profiles_per_company: 3
```

**Use case**: Complete business profiles with executive information
**Cost**: ~200-500 credits for 50 businesses

### 3. High-Volume Processing

```yaml
apify:
  enabled: true
  max_addresses: 500
  google_places:
    enabled: true
    max_places_per_search: 3  # Reduced to control costs
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 300
  linkedin_premium:
    enabled: false  # Too expensive for high volume
```

**Use case**: Large-scale contact discovery
**Cost**: ~500-1000 credits for 500 businesses

## Configuration Examples

### Industry-Specific Configurations

#### Legal Services (NAF 6920Z)
```yaml
apify:
  enabled: true
  max_addresses: 100
  google_places:
    enabled: true
    max_places_per_search: 8
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 100
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 50  # High value for legal professionals
    max_profiles_per_company: 5
```

#### Accounting Firms (NAF 6920Z)
```yaml
apify:
  enabled: true
  max_addresses: 200
  google_places:
    enabled: true
    max_places_per_search: 5
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 150
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 30
    max_profiles_per_company: 2
```

#### Consulting Services
```yaml
apify:
  enabled: true
  max_addresses: 75
  google_places:
    enabled: true
    max_places_per_search: 10
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 75
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 40
    max_profiles_per_company: 4
```

### Budget-Based Configurations

#### Low Budget (< 100 credits)
```yaml
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
    enabled: false
```

#### Medium Budget (100-300 credits)
```yaml
apify:
  enabled: true
  max_addresses: 75
  google_places:
    enabled: true
    max_places_per_search: 8
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 60
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 15
    max_profiles_per_company: 2
```

#### High Budget (300+ credits)
```yaml
apify:
  enabled: true
  max_addresses: 150
  google_places:
    enabled: true
    max_places_per_search: 10
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 120
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 50
    max_profiles_per_company: 5
```

## Cost Management

### Understanding Costs

| Scraper | Typical Cost per Operation | Data Quality |
|---------|---------------------------|--------------|
| Google Places | 1-5 credits per search | High accuracy for basic info |
| Maps Contacts | 1-3 credits per enrichment | Excellent for contact details |
| LinkedIn Premium | 10-50 credits per search | Premium executive data |

### Cost Control Strategies

#### 1. Address Quality Optimization
```yaml
# Improve address quality before processing
filters:
  regions: ["75", "92", "93"]  # Limit to specific regions
  active_only: true            # Only active businesses
```

#### 2. Smart Batching
```yaml
apify:
  max_addresses: 100          # Process in smaller batches
  google_places:
    max_places_per_search: 5  # Reduce results per search
```

#### 3. Progressive Enhancement
```yaml
# Start with basic enrichment
apify:
  google_places:
    enabled: true
  google_maps_contacts:
    enabled: true
  linkedin_premium:
    enabled: false  # Add later if budget allows
```

### Budget Monitoring

Monitor costs during execution:

```bash
# Enable cost tracking
python builder_cli.py run-profile \
  --job jobs/cost_monitored.yaml \
  --input data.parquet \
  --out out/monitored \
  --profile standard \
  --debug  # Shows cost estimates
```

## Best Practices

### 1. Data Preparation

Ensure high-quality input data:
```python
# Clean addresses before processing
df['adresse_complete'] = df['numero_voie'].astype(str) + ' ' + \
                        df['type_voie'].fillna('') + ' ' + \
                        df['libelle_voie'].fillna('') + ', ' + \
                        df['ville'].fillna('')
```

### 2. Quality Validation

Validate results after processing:
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

### 3. Error Handling

Implement robust error handling:
```yaml
apify:
  enabled: true
  retry_failed: true
  continue_on_error: true
  timeout_minutes: 30
```

### 4. Performance Optimization

Optimize for better performance:
```yaml
apify:
  max_addresses: 50           # Smaller batches for stability
  parallel_requests: 2        # Conservative parallelism
  request_delay: 1            # Delay between requests
```

## Integration Patterns

### 1. Pipeline Integration

Standard integration in processing profiles:
```yaml
# In job configuration
profile: "standard"  # Automatically includes Apify enrichment

# Custom profile with Apify
steps: [
  "dumps.collect",
  "normalize.standardize",
  "enrich.address",
  "api.apify",           # Apify enrichment
  "quality.score",
  "package.export"
]
```

### 2. Workflow Integration

Custom workflow with Apify:
```python
from api.apify_agents import run as run_apify

# Custom workflow
def custom_apify_workflow(input_file, output_dir):
    # Prepare context
    ctx = {
        'outdir_path': Path(output_dir),
        'logger': logger
    }
    
    # Apify configuration
    config = {
        'apify': {
            'enabled': True,
            'max_addresses': 100,
            'google_places': {'enabled': True},
            'google_maps_contacts': {'enabled': True},
            'linkedin_premium': {'enabled': True}
        }
    }
    
    # Run Apify enrichment
    result = run_apify(config, ctx)
    return result
```

### 3. API Integration

Direct API usage:
```python
from apify_client import ApifyClient

# Initialize client
client = ApifyClient(os.getenv('APIFY_API_TOKEN'))

# Run specific actor
run = client.actor('compass/crawler-google-places').call(
    run_input={
        'searchStringsArray': ['Restaurant Paris'],
        'maxCrawledPlacesPerSearch': 10
    }
)

# Get results
results = list(client.dataset(run['defaultDatasetId']).iterate_items())
```

---

For troubleshooting and advanced configuration, see [Apify Troubleshooting Guide](apify-troubleshooting.md).