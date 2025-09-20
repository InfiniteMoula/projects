# Apify Integration Documentation

This document explains how to configure and use the Apify platform integration in the data scraping and enrichment pipeline.

## 📚 Complete Documentation Suite

This is part of a comprehensive Apify documentation suite. For detailed information, see:

- **[📋 Complete Guide](./apify-complete-guide.md)** - Master index and overview
- **[🔧 Implementation Details](./apify-implementation-details.md)** - Technical architecture and code details  
- **[📖 Usage Guide](./apify-usage-guide.md)** - Practical usage patterns and examples
- **[🚀 Automation Roadmap](./apify-automation-roadmap.md)** - Future automation strategy
- **[🛠️ Troubleshooting](./apify-troubleshooting.md)** - Complete troubleshooting guide

---

## Overview

The Apify integration (`api.apify` step) enriches business data using three powerful Apify scrapers:

1. **Google Places Crawler** (compass/crawler-google-places)
2. **Google Maps with Contact Details** (lukaskrivka/google-maps-with-contact-details)  
3. **LinkedIn Premium Actor** (bebity/linkedin-premium-actor)

The integration reads addresses from step 7 (address extraction) and uses these scrapers to gather comprehensive business information including contact details, executive information, and business ratings.

## Setup

### 1. Get Apify API Token

1. Sign up or log in to [Apify Console](https://console.apify.com/)
2. Go to [Account Integrations](https://console.apify.com/account/integrations)
3. Copy your API token

### 2. Configure Environment

Add your API token to `.env` file:

```bash
# Apify API Configuration
APIFY_API_TOKEN=your_apify_api_token_here
```

### 3. Update Requirements

The Apify client is automatically included in `requirements.txt`:

```
apify-client==2.1.0
```

## Configuration

Add the `apify` configuration section to your job YAML file:

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

### Configuration Options

#### Main Configuration

- **`enabled`** (boolean): Enable/disable all Apify agents
- **`max_addresses`** (integer): Maximum number of addresses to process (cost control)
- **`save_raw_results`** (boolean): Save raw JSON results for debugging

#### Google Places Crawler

- **`enabled`** (boolean): Enable/disable Google Places scraping
- **`max_places_per_search`** (integer): Maximum places to return per address search

#### Google Maps Contact Details

- **`enabled`** (boolean): Enable/disable contact detail enrichment
- **`max_contact_enrichments`** (integer): Maximum places to enrich with contact details

#### LinkedIn Premium Actor

- **`enabled`** (boolean): Enable/disable LinkedIn executive search
- **`max_linkedin_searches`** (integer): Maximum companies to search on LinkedIn
- **`max_profiles_per_company`** (integer): Maximum executive profiles per company

## Pipeline Integration

### Step Dependencies

The Apify step depends on:
- **Step 7** (`enrich.address`): Address extraction that creates `database.csv` with addresses

### Processing Flow

1. **Address Input**: Reads addresses from `database.csv` (created by step 7)
2. **Google Places Search**: Searches each address for business information
3. **Contact Enrichment**: Enriches found places with detailed contact info
4. **LinkedIn Search**: Searches for company executives on LinkedIn
5. **Data Merge**: Combines all results into enriched dataset

### Output Data

The step creates `apify_enriched.parquet` with these additional columns:

#### Business Information
- `apify_places_found`: Number of places found for the address
- `apify_business_names`: Business names found (up to 3)
- `apify_phones`: Phone numbers found (up to 3)
- `apify_emails`: Email addresses found (up to 3)
- `apify_websites`: Website URLs found (up to 3)
- `apify_ratings`: Business ratings/scores (up to 3)
- `apify_categories`: Business categories (up to 3)

#### Executive Information
- `apify_executives`: Executive names and positions (up to 5)
- `apify_linkedin_profiles`: LinkedIn profile URLs (up to 5)

## Usage Examples

### Basic Usage

Run with standard profile (includes Apify step):

```bash
python builder_cli.py run-profile \
  --job examples/apify_job_example.yaml \
  --input data/sirene.parquet \
  --out out/apify_test \
  --profile standard
```

### Run Only Apify Step

```bash
python builder_cli.py run-step \
  --job examples/apify_job_example.yaml \
  --input data/sirene.parquet \
  --out out/apify_test \
  --step api.apify
```

### Cost-Controlled Testing

For testing with minimal costs:

```yaml
apify:
  enabled: true
  max_addresses: 5  # Process only 5 addresses
  google_places:
    enabled: true
    max_places_per_search: 2
  google_maps_contacts:
    enabled: false  # Disable to save costs
  linkedin_premium:
    enabled: false  # Disable to save costs
```

## Cost Management

### Apify Credits

Each scraper consumes Apify credits:
- **Google Places Crawler**: ~1-5 credits per search
- **Google Maps Contact Details**: ~1-3 credits per enrichment
- **LinkedIn Premium Actor**: ~10-50 credits per search

### Cost Control Options

1. **Limit addresses**: Use `max_addresses` to control total processing
2. **Disable expensive scrapers**: Set `enabled: false` for LinkedIn
3. **Reduce results**: Lower `max_places_per_search` and `max_profiles_per_company`
4. **Test mode**: Start with 1-2 addresses to estimate costs

## Troubleshooting

### Common Issues

#### "APIFY_API_TOKEN environment variable is required"
- Solution: Add your API token to `.env` file

#### "No input data found"
- Solution: Ensure step 7 (address extraction) has run and created `database.csv`

#### "Client error" or API failures
- Check your API token is valid
- Verify you have sufficient Apify credits
- Check network connectivity

#### High costs
- Reduce `max_addresses` setting
- Disable LinkedIn scraper which is most expensive
- Use smaller `max_places_per_search` values

### Debug Mode

Enable debug output with:

```bash
python builder_cli.py run-profile \
  --job examples/apify_job_example.yaml \
  --input data/sirene.parquet \
  --out out/apify_test \
  --profile standard \
  --debug
```

### Raw Results

When `save_raw_results: true`, raw JSON results are saved to:
- `apify_raw_results.json`: Contains all raw scraper responses

## Performance

### Typical Performance
- **Processing time**: 30-60 seconds per address (depending on results)
- **Memory usage**: Low to moderate
- **API rate limits**: Handled automatically by Apify platform

### Optimization Tips
1. **Batch processing**: Process in smaller batches for large datasets
2. **Profile selection**: Use 'standard' profile for balanced enrichment
3. **Cache results**: Avoid re-processing same addresses
4. **Monitor credits**: Check Apify console for credit usage

## Integration with Other Steps

The Apify enrichment integrates well with other pipeline steps:

- **Before**: Depends on address extraction (step 7)
- **After**: Results can be used by email/phone validation steps
- **Export**: Enriched data is included in final export files

## Security

- **API Token**: Keep your Apify API token secure and don't commit it to version control
- **Rate Limits**: Respect Apify platform rate limits
- **Data Privacy**: Be aware of data privacy laws when scraping business information