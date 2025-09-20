# Scraping Methods & Data Sources Guide

Comprehensive guide to all data collection and enrichment methods available in the business intelligence pipeline, covering web scraping, APIs, document processing, and specialized enrichment techniques.

## Table of Contents

1. [Overview](#overview)
2. [Web Scraping Methods](#web-scraping-methods)
3. [API Integration](#api-integration)
4. [Document Processing](#document-processing)
5. [Google Maps Integration](#google-maps-integration)
6. [Address-Based Enrichment](#address-based-enrichment)
7. [Data Enrichment Methods](#data-enrichment-methods)
8. [Configuration and Setup](#configuration-and-setup)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The pipeline supports multiple data collection methods that can be combined for comprehensive business intelligence gathering:

- **🌐 Web Scraping**: Static HTML, JavaScript-heavy sites, sitemaps
- **📡 API Integration**: External business data APIs and services
- **📄 Document Processing**: PDF extraction and content analysis
- **🗺️ Google Maps**: Business discovery and contact enrichment
- **📍 Address Search**: Location-based business intelligence
- **🔍 Data Enrichment**: Domain, email, phone validation and discovery

## Web Scraping Methods

### Static Web Scraping (`http.static`)

Collects data from static HTML pages with rate limiting and respect for robots.txt.

**Configuration:**
```yaml
http:
  seeds: ["https://example.com"]
  per_domain_rps: 0.5  # Requests per second
  user_agent: "BusinessIntelligenceBot/1.0"
  timeout: 30
  max_retries: 3
```

**Use Cases:**
- Company websites and contact pages
- Professional service directories
- Industry-specific listings

**Example:**
```yaml
http:
  seeds: 
    - "https://www.experts-comptables.fr/annuaire"
    - "https://www.ordre-des-avocats.fr/directory"
  per_domain_rps: 0.3
  timeout: 45
```

### Sitemap Discovery (`http.sitemap`)

Automatically discovers URLs from website sitemaps for comprehensive coverage.

**Configuration:**
```yaml
sitemap:
  domains: ["example.com"]
  allow_patterns: ["contact", "about", "team"]
  deny_patterns: ["blog", "news"]
  max_urls: 500
  timeout: 60
```

**Features:**
- Automatic sitemap.xml discovery
- Pattern-based URL filtering
- Respect for robots.txt directives
- Efficient bulk URL collection

**Example:**
```yaml
sitemap:
  domains: 
    - "experts-comptables.org"
    - "ordre-des-avocats.fr"
  allow_patterns: ["cabinet", "contact", "presentation"]
  max_urls: 1000
```

### Headless Browser Scraping (`headless.collect`)

Handles JavaScript-heavy websites and dynamic content using automated browsers.

**Features:**
- Full JavaScript rendering
- Dynamic content extraction
- Form interaction capabilities
- Screenshot capture for debugging

**Configuration:**
```yaml
headless:
  browser: "chromium"
  timeout: 60
  wait_for: "networkidle"
  viewport: {"width": 1920, "height": 1080}
  screenshots: true
```

**Use Cases:**
- Single Page Applications (SPAs)
- JavaScript-rendered contact forms
- Dynamic business directories
- AJAX-loaded content

## API Integration

### External Business APIs (`api.collect`)

Integrates with external business data APIs for additional enrichment.

**Configuration:**
```yaml
api:
  endpoints:
    - name: "business_directory"
      url: "https://api.business-directory.com/v1/search"
      headers:
        Authorization: "Bearer ${API_KEY}"
      params:
        query: "{company_name}"
        location: "{city}"
```

**Supported API Types:**
- Business directory APIs
- Contact validation services
- Industry-specific databases
- Government business registries

### Hunter.io Email Validation

Professional email validation and discovery service.

**Setup:**
```bash
# In .env file
HUNTER_API_KEY=your_hunter_api_key_here
```

**Features:**
- Email format validation
- Domain verification
- Deliverability scoring
- Professional email discovery

## Document Processing

### PDF Document Collection (`pdf.collect`)

Downloads and processes PDF documents for business information extraction.

**Configuration:**
```yaml
pdf:
  urls: ["https://example.com/brochure.pdf"]
  max_size_mb: 10
  timeout: 60
  extract_text: true
  extract_metadata: true
```

**Use Cases:**
- Company brochures and presentations
- Annual reports and financial documents
- Professional service portfolios
- Legal documentation

### PDF Content Extraction (`parse.pdf`)

Extracts structured information from PDF documents.

**Features:**
- Text extraction with layout preservation
- Metadata extraction (author, creation date, etc.)
- Contact information discovery
- Professional qualification extraction

**Example Extracted Data:**
- Company contact information
- Professional certifications
- Service descriptions
- Executive names and titles

### RSS/Atom Feed Processing (`feeds.collect`)

Processes RSS and Atom feeds for business news and updates.

**Configuration:**
```yaml
feeds:
  urls: ["https://example.com/feed.xml"]
  max_entries: 50
  timeout: 30
  extract_contacts: true
```

**Use Cases:**
- Industry news and announcements
- Company press releases
- Professional blog content
- Event announcements

## Google Maps Integration

### Business Discovery and Enrichment

Comprehensive Google Maps integration for business information gathering.

**Data Extracted:**
- Phone numbers and contact details
- Website URLs and social media links
- Business hours and location information
- Customer ratings and review counts
- Business categories and descriptions
- Director/manager names

**Configuration:**
```yaml
google_maps:
  enabled: true
  max_results_per_business: 3
  timeout_seconds: 30
  search_patterns:
    - "{denomination} {ville}"
    - "{denomination} {code_postal}"
    - "SIREN {siren}"
  requests_per_second: 0.5
```

**Search Strategies:**
1. **Primary Search**: `{company_name} {city}`
2. **Postal Code Search**: `{company_name} {postal_code}`
3. **SIREN Search**: `SIREN {business_id}`
4. **Fallback Search**: `{company_name} France`

**Quality Validation:**
- French phone number format validation
- Email format and domain verification
- Website URL accessibility checks
- Address format validation

## Address-Based Enrichment

### Address Search Enhancement

Searches Google.fr and Bing.com for business addresses to discover additional information.

**Process:**
1. Extracts unique addresses from normalized dataset
2. Searches multiple search engines for each address
3. Parses search results to extract business information
4. Merges discovered data back into main dataset

**Configuration:**
```yaml
address_search:
  timeout: 10.0
  request_delay: [1.0, 2.5]  # Random delay range
  max_workers: 2
  retry_count: 2
  search_engines: ["google.fr", "bing.com"]
```

**Extracted Information:**
- Business names at specific addresses
- Phone numbers associated with locations
- Email addresses for businesses
- Additional company information

**Output Fields:**
- `found_business_names_str`: Business names (separated by '; ')
- `found_phones_str`: Phone numbers (separated by '; ')
- `found_emails_str`: Email addresses (separated by '; ')
- `search_status`: Search result status

**Rate Limiting:**
- Random delays between requests (1-2.5 seconds)
- Limited concurrency (2 parallel searches)
- Proper user-agent identification
- Retry logic for failed requests

## Data Enrichment Methods

### Domain Discovery (`enrich.domain`)

Discovers and validates business domains and websites.

**Process:**
1. Extract domains from existing website data
2. Generate potential domain names from company names
3. Validate domain existence and accessibility
4. Score domain relevance and quality

**Techniques:**
- Domain name generation heuristics
- DNS validation checks
- HTTP accessibility testing
- Domain reputation scoring

### Email Discovery (`enrich.email`)

Intelligent email address discovery and validation.

**Strategies:**
1. **Pattern-based generation**: `contact@domain.com`, `info@domain.com`
2. **Heuristic combinations**: `firstname.lastname@domain.com`
3. **Website extraction**: Parse contact pages for email addresses
4. **External validation**: Hunter.io integration for verification

**Configuration:**
```yaml
enrich:
  email_formats_priority: 
    - "contact@{domain}"
    - "info@{domain}"
    - "{firstname}.{lastname}@{domain}"
    - "commercial@{domain}"
```

### Phone Number Processing (`enrich.phone`)

French phone number normalization and validation.

**Features:**
- Format standardization (international vs. national)
- Number validation and verification
- Mobile vs. landline classification
- Geographic region identification

**Supported Formats:**
- International: `+33 1 42 86 87 88`
- National: `01 42 86 87 88`
- Various separators: spaces, dots, dashes

### DNS Validation (`enrich.dns`)

Domain Name System validation and analysis.

**Checks Performed:**
- Domain resolution verification
- MX record validation (email capability)
- A/AAAA record verification (website accessibility)
- DNS response time measurement

### Website Probing (`enrich.site`)

Website accessibility and technology analysis.

**Features:**
- HTTP/HTTPS accessibility testing
- Response time measurement
- Technology stack detection
- Content type analysis
- Redirect chain following

## Configuration and Setup

### Environment Variables

```bash
# API Keys
HUNTER_API_KEY=your_hunter_api_key_here

# Proxy Configuration (optional)
HTTP_PROXY=http://proxy:port
HTTPS_PROXY=https://proxy:port

# User Agent Configuration
USER_AGENT="BusinessIntelligenceBot/1.0"
```

### Job Configuration Template

```yaml
niche: "comprehensive_scraping"

# Web scraping configuration
http:
  seeds: 
    - "https://www.experts-comptables.fr"
    - "https://www.ordre-des-avocats.fr"
  per_domain_rps: 0.5
  timeout: 30

# Sitemap processing
sitemap:
  domains: ["professionals.org", "business-directory.com"]
  allow_patterns: ["contact", "about", "services"]
  max_urls: 1000

# Document processing
pdf:
  urls: ["https://example.com/directory.pdf"]
  max_size_mb: 10

# Feed processing
feeds:
  urls: ["https://industry-news.com/feed.xml"]
  max_entries: 100

# Google Maps integration
google_maps:
  enabled: true
  max_results_per_business: 5
  timeout_seconds: 45

# Enrichment configuration
enrich:
  email_formats_priority: 
    - "contact@{domain}"
    - "info@{domain}"
    - "commercial@{domain}"

# Resource budgets
budgets:
  max_http_requests: 2000
  max_http_bytes: 52428800  # 50MB
  time_budget_min: 120
  ram_mb: 4096
```

## Best Practices

### 1. Respectful Scraping

```yaml
http:
  per_domain_rps: 0.5      # Conservative rate limiting
  respect_robots_txt: true  # Honor robots.txt
  user_agent: "BusinessIntelligenceBot/1.0"
  timeout: 30              # Reasonable timeouts
```

### 2. Error Handling

```yaml
http:
  max_retries: 3
  retry_delays: [1, 2, 5]  # Exponential backoff
  continue_on_error: true
```

### 3. Quality Control

Implement validation for extracted data:
- Phone number format validation
- Email format verification
- Domain accessibility checks
- Data completeness scoring

### 4. Performance Optimization

```yaml
# Parallel processing
http:
  max_workers: 4  # Adjust based on available resources

# Memory management
budgets:
  ram_mb: 4096    # Set appropriate memory limits

# Request optimization
http:
  timeout: 30     # Balance between completeness and speed
```

### 5. Legal Compliance

- Always respect robots.txt directives
- Implement reasonable rate limiting
- Use identifiable user agents
- Avoid scraping behind logins or paywalls
- Respect copyright and terms of service

## Troubleshooting

### Common Issues

#### 1. Rate Limiting / HTTP 429
**Problem**: Getting rate-limited by websites
**Solution**: 
```yaml
http:
  per_domain_rps: 0.2  # Reduce rate
  request_delay: 5     # Add delays
```

#### 2. JavaScript Content Not Loading
**Problem**: Missing content from dynamic sites
**Solution**: Use headless browser collection
```yaml
# Enable headless collection for dynamic content
profiles:
  standard: 
    steps: [..., "headless.collect", ...]
```

#### 3. Memory Issues
**Problem**: High memory usage during processing
**Solution**: 
```yaml
budgets:
  ram_mb: 2048      # Set memory limits
http:
  max_workers: 2    # Reduce parallelism
```

#### 4. No Results from Search Engines
**Problem**: Address searches returning empty results
**Solution**: 
- Verify network connectivity
- Check for IP blocking
- Reduce request frequency
- Improve address quality

#### 5. PDF Processing Failures
**Problem**: PDF documents not processing correctly
**Solution**:
```yaml
pdf:
  max_size_mb: 20    # Increase size limit
  timeout: 120       # Increase timeout
  fallback_ocr: true # Enable OCR for scanned PDFs
```

### Debug Commands

```bash
# Test HTTP scraping
python builder_cli.py run-step --step http.static --job test.yaml --debug

# Test Google Maps integration
python builder_cli.py run-step --step enrich.google_maps --job test.yaml --debug

# Test address search
python builder_cli.py run-step --step enrich.address --job test.yaml --debug

# Full debug run
python builder_cli.py run-profile --job jobs/debug.yaml --profile standard --debug --verbose --sample 5
```

### Performance Monitoring

Monitor scraping performance:
```bash
# Check success rates
grep "HTTP" out/logs/run_*.jsonl | grep -E "(200|404|429)"

# Monitor memory usage
grep "RAM" out/logs/run_*.jsonl

# Check processing speed
grep "processed" out/logs/run_*.jsonl
```

---

For Apify-specific scraping methods, see [Apify Setup & Usage Guide](apify-setup-guide.md).
For troubleshooting advanced issues, see the main [README troubleshooting section](../README.md#troubleshooting).