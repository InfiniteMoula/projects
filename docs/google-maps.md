# Google Maps Integration Guide

## Overview

This guide covers the Google Maps integration features in the data scraping and enrichment pipeline, including setup, configuration, and usage patterns.

## Features

### Google Maps Search Integration

The Google Maps enrichment module (`enrich.google_maps_search`) provides comprehensive business data extraction from Google Maps listings.

### Data Extracted

The module adds 9 new columns to your dataset:

| Column | Description | Example |
|--------|-------------|---------|
| `gm_phone` | Phone number from Google Maps | "+33 1 42 86 87 88" |
| `gm_website` | Website URL from Google Maps | "https://www.example.com" |
| `gm_email` | Email address discovered | "contact@example.com" |
| `gm_hours` | Business hours | "Lun-Ven 9h-18h" |
| `gm_rating` | Customer rating (1-5 stars) | "4.2" |
| `gm_reviews` | Number of reviews | "156" |
| `gm_address` | Confirmed address | "123 Rue de la Paix, 75001 Paris" |
| `gm_category` | Business category | "Expert-comptable" |
| `gm_director` | Director/manager name | "Jean Dupont" |

## Integration in Pipeline

### Step Registration

The Google Maps enrichment is registered as a pipeline step:

```python
"enrich.google_maps": "enrich.google_maps_search:run"
```

### Dependencies

The step depends on data normalization:

```python
"enrich.google_maps": {"normalize.standardize"}
```

### Profile Integration

- **Standard Profile**: Includes Google Maps enrichment
- **Deep Profile**: Includes Google Maps enrichment  
- **Quick Profile**: Excludes Google Maps (for faster processing)

## Configuration

### Job Configuration

Add Google Maps settings to your job YAML:

```yaml
# Google Maps enrichment settings
google_maps:
  enabled: true
  max_results_per_business: 3
  timeout_seconds: 30
  search_patterns:
    - "{denomination} {ville}"
    - "{denomination} {code_postal}"
    - "SIREN {siren}"
  
# Increased budgets for Google Maps
budgets:
  max_http_requests: 2000    # Increased for Maps searches
  max_http_bytes: 52428800   # 50MB for Maps content
  time_budget_min: 90        # Extended time budget
  ram_mb: 4096              # 4GB RAM for processing
```

### Rate Limiting

Google Maps searches respect rate limiting:

```yaml
google_maps:
  requests_per_second: 0.5   # Conservative rate limit
  retry_delays: [1, 2, 5]    # Exponential backoff
  max_retries: 3
```

## Usage Examples

### Basic Google Maps Enrichment

```bash
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_latest.parquet \
  --out out/experts_with_maps \
  --profile standard \
  --verbose
```

### Address Search Focus

For businesses needing accurate address validation:

```bash
python builder_cli.py run-step \
  --step enrich.google_maps \
  --job jobs/address_focused.yaml \
  --input data/normalized.parquet \
  --out out/address_enriched \
  --verbose
```

### Batch Processing with Maps

```bash
python builder_cli.py batch \
  --naf 6920Z --naf 6910Z \
  --input data/sirene.parquet \
  --output-dir out/professional_services \
  --profile standard \
  --verbose
```

## Search Strategies

### Multi-Pattern Search

The module uses multiple search patterns for better results:

1. **Primary Search**: `{denomination} {ville}`
2. **Postal Code Search**: `{denomination} {code_postal}`  
3. **SIREN Search**: `SIREN {siren}`
4. **Fallback Search**: `{denomination} France`

### Data Extraction Patterns

#### Phone Number Extraction
```python
phone_patterns = [
    r'(\+33\s?[1-9](?:\s?\d{2}){4})',
    r'(0[1-9](?:\s?\d{2}){4})',
    r'(\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2})'
]
```

#### Email Extraction
```python
email_patterns = [
    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    r'contact@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    r'info@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
]
```

#### Director Name Extraction
```python
director_patterns = [
    r'(?:Dirigeant|Gérant|Président|Directeur|Manager):\s*([^,\n]+)',
    r'(?:dirigé par|géré par|sous la direction de)\s+([^,\n]+)'
]
```

## Quality and Validation

### Data Quality Checks

The module includes validation for extracted data:

- **Phone Validation**: French phone number format validation
- **Email Validation**: Email format and domain checks
- **URL Validation**: Website URL accessibility checks
- **Address Validation**: Address format and postal code validation

### Confidence Scoring

Each extracted field includes a confidence score:

```python
confidence_scores = {
    'gm_phone': 0.95,      # High confidence for formatted phones
    'gm_email': 0.80,      # Medium confidence for pattern-matched emails
    'gm_address': 0.90,    # High confidence for structured addresses
    'gm_director': 0.70    # Lower confidence for name extraction
}
```

## Performance Optimization

### Memory Management

- Processes businesses in batches to control memory usage
- Implements cleanup after each batch
- Monitors memory usage throughout processing

### Request Optimization

- Uses session pooling for HTTP requests
- Implements intelligent caching
- Respects robots.txt and rate limits

### Error Handling

- Graceful degradation when Google Maps is unavailable
- Retry logic with exponential backoff
- Detailed error logging for troubleshooting

## Monitoring and Debugging

### Debug Mode

Enable detailed logging for Google Maps operations:

```bash
python builder_cli.py run-profile \
  --job jobs/debug_maps.yaml \
  --input data/sample.parquet \
  --out out/debug \
  --profile standard \
  --debug \
  --sample 10
```

### Performance Metrics

Monitor Google Maps enrichment performance:

- **Search Success Rate**: Percentage of successful business lookups
- **Data Extraction Rate**: Percentage of fields successfully extracted
- **Average Response Time**: Time per Google Maps search
- **Error Rate**: Percentage of failed requests

### Common Issues and Solutions

#### Rate Limiting
**Problem**: Getting rate-limited by Google Maps
**Solution**: Reduce `requests_per_second` or increase delays

#### No Results Found
**Problem**: Businesses not found in Google Maps
**Solution**: Adjust search patterns or add fallback strategies

#### Extraction Failures
**Problem**: Data extraction returning empty results
**Solution**: Update extraction patterns or review HTML structure changes

## Legal and Compliance

### Robots.txt Compliance
- Respects robots.txt directives
- Implements polite crawling practices
- Includes User-Agent identification

### Terms of Service
- Follows Google's Terms of Service
- Implements reasonable request rates
- Avoids automated interactions beyond data collection

### Data Usage
- Extracted data is for legitimate business purposes
- No storage of proprietary Google data
- Compliance with GDPR and data protection regulations

## Integration with Other Features

### Address Search Module
The Google Maps integration works with the address search functionality for enhanced location validation.

### Quality Scoring
Google Maps data contributes to overall quality scores:
- Contact information completeness
- Address validation confidence  
- Business information freshness

### Export Integration
Google Maps fields are included in all export formats:
- CSV exports with GM_ prefixed columns
- Parquet exports with full metadata
- Quality reports with Maps-specific metrics

For more information, see the main [README.md](../README.md) documentation.