# Address Search Enrichment

This module adds address-based business information enrichment to the data pipeline. It searches Google.fr and Bing.com for business addresses to discover additional business names, phone numbers, and email addresses.

## Overview

The address search functionality:
1. Extracts addresses from the normalized dataset
2. Searches Google.fr and Bing.com for each unique address
3. Parses search results to extract business information
4. Merges found data back into the main dataset

## Pipeline Integration

The address search step (`enrich.address`) is automatically included in:
- **Standard profile**: Runs after phone enrichment
- **Deep profile**: Runs after phone enrichment

### Dependencies
- Requires `normalize.standardize` step to have completed
- Input: `normalized.parquet` or `normalized.csv`
- Output: `address_enriched.parquet`

## Configuration

### Default Settings
```python
SEARCH_TIMEOUT = 10.0  # HTTP timeout in seconds
REQUEST_DELAY = (1.0, 2.5)  # Random delay between requests
MAX_WORKERS = 2  # Parallel search threads
RETRY_COUNT = 2  # Number of retries for failed requests
```

### Search Engines
- **Google.fr**: `https://www.google.fr/search?q={address}`
- **Bing.com**: `https://www.bing.com/search?q={address}`

## Data Schema

### Input Requirements
The input dataset must contain an `adresse` column with business addresses.

### Output Fields
The enrichment adds these columns to the dataset:

| Column | Type | Description |
|--------|------|-------------|
| `found_business_names_str` | string | Business names found (separated by '; ') |
| `found_phones_str` | string | Phone numbers found (separated by '; ') |
| `found_emails_str` | string | Email addresses found (separated by '; ') |
| `search_status` | string | Search result status |

### Status Values
- `success`: Search completed successfully
- `partial_error_google`: Google search failed
- `partial_error_bing`: Bing search failed
- `error`: Both searches failed
- `not_searched`: Address was skipped

## Usage Examples

### Manual Execution
```python
from enrich.address_search import run as run_address_search

ctx = {
    'outdir': '/path/to/output',
    'logger': logger  # optional
}

result = run_address_search({}, ctx)
print(f"Status: {result['status']}")
print(f"Addresses searched: {result['addresses_searched']}")
```

### CLI Integration
```bash
# Run with standard profile (includes address search)
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data.parquet \
  --out output \
  --profile standard

# Run only address search step
python builder_cli.py run-step \
  --step enrich.address \
  --outdir /path/to/working/directory
```

## Rate Limiting & Ethics

The module implements several measures to be respectful to search engines:

1. **Random delays**: 1-2.5 seconds between requests
2. **Limited concurrency**: Maximum 2 parallel searches
3. **Proper user-agent**: Identifies as a regular browser
4. **Retry logic**: Handles temporary failures gracefully
5. **Timeout protection**: Prevents hanging requests

## Extraction Logic

### Phone Numbers
Extracts French phone number patterns:
- International format: `+33 X XX XX XX XX`
- National format: `0X XX XX XX XX`
- Various separators: spaces, dots, dashes

### Email Addresses
Standard email regex pattern:
- Format: `name@domain.tld`
- Validates basic email structure

### Business Names
Heuristic extraction from search results:
- HTML headers (h1-h4 tags)
- Elements containing business keywords (SA, SARL, Cabinet, etc.)
- Limited to reasonable length (20-200 characters)
- Removes duplicates and limits to top 5 results

## Performance

### Typical Performance
- **Addresses per minute**: ~30-60 (depending on search engine response)
- **Memory usage**: Low (streaming processing)
- **Network usage**: ~2-4 requests per address

### Optimization Tips
1. **Filter addresses**: Remove duplicates before processing
2. **Batch processing**: Process in smaller chunks for large datasets
3. **Monitor rate limits**: Watch for HTTP 429 responses
4. **Cache results**: Consider caching for repeated runs

## Error Handling

### Common Issues
1. **HTTP 429 (Rate Limited)**: Increase delays, reduce concurrency
2. **HTTP 403 (Blocked)**: Change user-agent, add delays
3. **Network timeouts**: Check connectivity, increase timeout
4. **No results**: Search engines may block automated requests

### Monitoring
Check the `search_status` column in output data:
```python
df = pd.read_parquet('address_enriched.parquet')
status_counts = df['search_status'].value_counts()
print(status_counts)
```

## Testing

Run the test suite:
```bash
cd /home/runner/work/projects/projects
PYTHONPATH=/home/runner/work/projects/projects python tests/test_address_search.py
```

Run the demonstration:
```bash
python /tmp/demo_address_search.py
```

## Troubleshooting

### No Results Found
This is common and can indicate:
- Search engines blocking automated requests
- Addresses not yielding business results
- Network connectivity issues

**Solutions:**
- Verify addresses are valid business locations
- Check network connectivity
- Monitor for rate limiting responses

### High Error Rates
**Symptoms:** Many addresses have `error` status

**Solutions:**
- Increase request delays
- Reduce concurrent workers
- Check user-agent string
- Verify search engine availability

### Performance Issues
**Symptoms:** Very slow processing

**Solutions:**
- Reduce timeout values
- Increase concurrency (carefully)
- Filter duplicate addresses
- Process in smaller batches