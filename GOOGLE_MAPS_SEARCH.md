# Google Maps Search Enrichment

## Overview

The Google Maps search enrichment module (`enrich.google_maps_search`) adds comprehensive business information by searching Google Maps using company addresses. This module extracts valuable company data including contact information, ratings, business types, and web presence.

## Features

### Data Extracted
- **Business Names**: Official business names from Google Maps listings
- **Phone Numbers**: Contact phone numbers in French format
- **Email Addresses**: Business email addresses found in listings
- **Business Types**: Categories and service types (e.g., "Cabinet", "Restaurant", "Entreprise")
- **Ratings**: Google Maps star ratings (1-5 scale)
- **Review Counts**: Number of customer reviews
- **Websites**: Business website URLs
- **Search Status**: Success/failure status for debugging

### Address Query Building

The module intelligently constructs search queries using available address components:

1. **Preferred Method**: Uses individual address components when available:
   - `numero_voie` + `type_voie` + `libelle_voie` + `commune` + `code_postal`

2. **Fallback Method**: Uses complete address field:
   - `adresse` + `commune` + `code_postal`

3. **Enhancement**: Adds company name (`raison_sociale`) for better search accuracy

## Pipeline Integration

### Registry Entry
- **Module**: `enrich.google_maps_search:run`
- **Input**: `normalized.parquet` or `normalized.csv`
- **Output**: `google_maps_enriched.parquet`

### Dependencies
- Requires normalized data with address information
- Works with existing standardization and quality modules

### Data Flow
```
normalized.parquet → google_maps_search → google_maps_enriched.parquet → package.export → dataset.csv
```

## Configuration

### Rate Limiting
- **Request Delay**: 2-4 seconds between requests
- **Max Workers**: 1 (sequential processing to avoid blocking)
- **Timeout**: 15 seconds per request
- **Retries**: 2 attempts per failed request

### HTTP Headers
Uses realistic browser headers to avoid detection:
- User-Agent: Chrome on Windows
- Accept headers for HTML content
- French language preference

## Output Columns

The enrichment adds the following columns to the dataset:

| Column | Type | Description |
|--------|------|-------------|
| `maps_business_names` | String | Business names found (semicolon-separated) |
| `maps_phone_numbers` | String | Phone numbers found (semicolon-separated) |
| `maps_emails` | String | Email addresses found (semicolon-separated) |
| `maps_business_types` | String | Business categories found (semicolon-separated) |
| `maps_websites` | String | Website URLs found (semicolon-separated) |
| `maps_rating` | Float | Google Maps rating (1.0-5.0) |
| `maps_review_count` | Integer | Number of reviews |
| `maps_search_status` | String | Search result status |

## Status Values

The `maps_search_status` column indicates the result of each search:

- `success`: Data successfully extracted
- `timeout`: Request timed out
- `http_error_XXX`: HTTP error with status code
- `empty_query`: No valid address to search
- `error_XXX`: Other errors with exception type
- `not_searched`: Row was not processed

## Usage Example

### Via Pipeline
```bash
python builder_cli.py run-profile --job jobs/example.yaml --out ./output --profile standard
```

### Direct Usage
```python
from enrich import google_maps_search

ctx = {
    "outdir": "./output",
    "logger": logger  # Optional
}

result = google_maps_search.run({}, ctx)
```

## Error Handling

The module handles various error conditions gracefully:

- **Missing Data**: Returns `SKIPPED` if no normalized data found
- **Empty Data**: Returns `SKIPPED` if input dataset is empty
- **Missing Columns**: Returns `SKIPPED` with specific missing column info
- **Network Errors**: Individual searches fail gracefully, overall process continues
- **Parsing Errors**: Failed extractions are logged, don't stop processing

## Performance Considerations

### Processing Time
- Approximately 3-5 seconds per unique address
- Sequential processing to respect rate limits
- Can take 15-30 minutes for datasets with 100+ unique addresses

### Memory Usage
- Processes all data in memory
- Reasonable for datasets up to 10,000 companies
- Consider batching for larger datasets

### Network Requirements
- Requires stable internet connection
- May be blocked by Google's anti-bot measures
- Success rate typically 70-90% depending on query quality

## Integration with Export

The Google Maps enrichment data is automatically included in:

1. **CSV Export**: All maps columns included in `dataset.csv`
2. **Parquet Export**: All maps columns included in `dataset.parquet`
3. **Quality Reports**: Maps data included in data quality assessments
4. **Data Dictionary**: Maps columns documented in metadata

## Best Practices

### Input Data Quality
- Ensure addresses are properly formatted and complete
- Verify company names are accurate for better search results
- Clean address data before running enrichment

### Rate Limiting Compliance
- Run during off-peak hours when possible
- Monitor for HTTP 429 (Too Many Requests) errors
- Consider using VPN or rotating IPs for large datasets

### Result Validation
- Check `maps_search_status` for successful searches
- Validate extracted phone numbers and emails
- Cross-reference with existing contact data

## Testing

Comprehensive test suite covers:
- Address query building with various input formats
- HTML parsing and data extraction
- Error handling scenarios
- Integration with pipeline components
- Mock HTTP responses for consistent testing

Run tests:
```bash
python -m pytest tests/test_google_maps_search.py -v
```

## Legal Considerations

- Respects robots.txt where applicable
- Uses rate limiting to minimize server load
- For research and business intelligence purposes
- Users responsible for compliance with Google's Terms of Service