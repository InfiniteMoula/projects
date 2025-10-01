# Business Intelligence Data Pipeline

A comprehensive business data intelligence pipeline that transforms raw business databases into enriched, actionable datasets. Specifically designed for French business data (SIRENE database), this system automatically discovers and enriches business information with contact details, executive information, and quality scoring through advanced web scraping and API integrations.

## What This Pipeline Does

**Transform raw business data into comprehensive business intelligence:**

- **Input**: French business database (SIRENE format) with basic company information
- **Process**: Multi-source data enrichment using web scraping, APIs, and AI-powered extraction
- **Output**: Complete business profiles with contact information, executive details, and quality scores

**Key Business Value:**
- 🏢 **Complete Business Profiles**: Company details, contact information, executive data
- 📞 **Contact Discovery**: Emails, phone numbers, websites from multiple sources  
- 👔 **Executive Information**: CEO, CFO, Directors via LinkedIn integration
- 🌐 **Multi-Source Enrichment**: Google Maps, LinkedIn, direct web scraping
- 📊 **Quality Control**: Automated scoring, validation, and deduplication
- 📈 **Scalable Processing**: Batch processing with budget and performance controls

## Table of Contents

- [What This Pipeline Does](#what-this-pipeline-does)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Processing Profiles](#processing-profiles)
- [Documentation](#documentation)
- [Command Line Interface](#command-line-interface)
- [Job Configuration](#job-configuration)
- [Pipeline Steps](#pipeline-steps)
- [Batch Processing](#batch-processing)
- [Budget and KPI System](#budget-and-kpi-system)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Features

- **🤖 Intelligent Data Enrichment**: Multi-source business intelligence gathering
- **🌐 Advanced Web Scraping**: Static pages, dynamic content, sitemaps, PDFs, RSS feeds
- **🔧 Apify Platform Integration**: Professional Google Places, Google Maps, and LinkedIn scrapers
- **📧 Contact Discovery**: Smart email discovery, phone normalization, website validation  
- **👔 Executive Intelligence**: CEO, CFO, Directors and Founder information via LinkedIn
- **🎯 Quality Control**: Comprehensive scoring, validation checks, and confidence ratings
- **⚡ Flexible Processing**: Quick, standard, and deep processing modes
- **💰 Budget Management**: Control resource usage (time, RAM, HTTP requests, API costs)
- **🔄 Batch Processing**: Process multiple business categories (NAF codes) efficiently
- **📊 Rich Exports**: CSV, Parquet, with interactive quality reports (HTML/PDF)
- **🔍 Comprehensive Logging**: Detailed execution tracking and debugging capabilities

## Quick Start

Get started in under 5 minutes with a sample dataset:

### 1. Install and Setup
```bash
# Clone and setup
git clone <repository-url>
cd projects
pip install -r requirements.txt

# Configure environment (optional for basic use)
cp .env.example .env
# Edit .env with your API keys (APIFY_API_TOKEN, HUNTER_API_KEY)
```

### 2. Run Your First Enrichment
```bash
# Quick test with sample data (no API keys required)
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_sample.parquet \
  --out out/first_test \
  --profile quick \
  --sample 10 \
  --dry-run
```

### 3. Full Processing Example
```bash
# Complete enrichment with all features
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_sample.parquet \
  --out out/experts_comptables \
  --profile standard
```

📋 **Output**: Find enriched data in `out/experts_comptables/dataset.csv` with complete business profiles!

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum (4GB recommended for standard/deep profiles)
- **Storage**: SSD recommended for better performance
- **Network**: Internet connection for web scraping and API access

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**: `pandas`, `httpx`, `beautifulsoup4`, `PyYAML`, `psutil`, `weasyprint`, `apify-client`

### Environment Setup (Optional)

For advanced features, configure API access:

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials:
# APIFY_API_TOKEN=your_apify_token_here     # For Apify platform scrapers
# HUNTER_API_KEY=your_hunter_key_here       # For email validation
# HTTP_PROXY=http://proxy:port              # Optional proxy settings
```

> **Note**: Basic web scraping works without API keys. Apify integration requires an [Apify account](https://console.apify.com/account/integrations).

## Processing Profiles

Choose the right processing profile based on your needs and available time:

### Quick Profile (⚡ 5-15 minutes)
**Best for**: Fast processing, basic enrichment, testing
```yaml
steps: ["dumps.collect", "api.collect", "normalize.standardize", "quality.checks", "quality.score", "package.export"]
```
- ✅ Basic data collection and normalization
- ✅ Quality scoring and validation  
- ✅ Fast turnaround for testing
- ❌ No web scraping or external APIs

### Standard Profile (🚀 20-60 minutes)  
**Best for**: Balanced processing with comprehensive enrichment
```yaml
steps: ["dumps.collect", "api.collect", "feeds.collect", "parse.jsonld", "normalize.standardize", 
        "enrich.address", "api.apify", "enrich.google_maps", "enrich.domain", "enrich.site", 
        "enrich.dns", "enrich.email", "enrich.phone", "quality.checks", "quality.score", "package.export"]
```
- ✅ Web scraping and sitemap discovery
- ✅ **Apify platform integration** (Google Maps, LinkedIn)
- ✅ Domain and email enrichment
- ✅ Contact discovery and validation
- ✅ Comprehensive quality control

### Deep Profile (🔍 1-3 hours)
**Best for**: Maximum data extraction, comprehensive analysis
```yaml
# Includes all Standard features plus:
# - Headless browser automation
# - PDF document processing  
# - Advanced HTML parsing
# - Extended quality analysis
```
- ✅ Everything from Standard profile
- ✅ JavaScript-heavy website processing
- ✅ PDF document extraction
- ✅ Advanced content parsing
- ✅ Maximum data completeness

## Documentation

### 📚 Complete Documentation Suite

- **[Apify Setup & Usage Guide](docs/apify-setup-guide.md)** - Complete Apify platform integration
- **[Scraping Methods Guide](docs/scraping-methods-guide.md)** - Web scraping, APIs, and data sources  
- **[Technical Implementation Details](docs/technical-implementation-details.md)** - Technical implementation details
- **[Apify Troubleshooting](docs/apify-troubleshooting.md)** - Advanced troubleshooting and optimization
- **[Apify Implementation Details](docs/apify-implementation-details.md)** - Deep technical architecture

### 🔧 Quick References

- **[Legal Compliance](docs/legal.md)** - Robots.txt, ToS compliance, data usage guidelines
- **[Package Export Documentation](package/README.md)** - Export formats and quality reporting

## Command Line Interface

### Main Commands

```bash
python builder_cli.py {run-step,run-profile,batch,resume}
```

### run-profile - Execute Complete Processing

```bash
python builder_cli.py run-profile [OPTIONS]
```

**Required Arguments:**
- `--job JOB`: Path to job configuration file
- `--out OUT`: Output directory
- `--profile {quick,standard,deep}`: Processing profile

**Common Options:**
- `--input INPUT`: Input data file (Parquet/CSV)
- `--dry-run`: Simulate execution without processing
- `--sample SAMPLE`: Limit processing to N records
- `--workers WORKERS`: Number of worker threads (default: 8)
- `--verbose`: Enable detailed logging
- `--debug`: Enable debug mode

### batch - Process Multiple Business Categories

```bash
python builder_cli.py batch --naf 6920Z --naf 4329A --input data.parquet --output-dir out/
```

### Examples

**Quick test:**
```bash
python builder_cli.py run-profile --job jobs/experts_comptables.yaml --input data.parquet --out out/test --profile quick --sample 10 --debug
```

**Production run:**
```bash
python builder_cli.py run-profile --job jobs/experts_comptables.yaml --input data.parquet --out out/production --profile standard
```

## Job Configuration

Jobs are configured using YAML files. Basic structure:

```yaml
niche: "job_name"

# Data filtering
filters:
  naf_include: ["6920Z"]  # Business category codes
  regions: ["75", "92"]   # Geographic regions (optional)

# Apify platform integration
apify:
  enabled: true
  max_addresses: 50
  google_places:
    enabled: true
    max_places_per_search: 10
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 20

# Quality targets
kpi_targets:
  min_quality_score: 80
  min_email_plausible_pct: 60

# Resource budgets
budgets:
  max_http_requests: 2000
  time_budget_min: 90
  ram_mb: 4096
```

## Pipeline Steps

The system processes data through these main stages:

### 1. Data Collection
- **dumps.collect**: Load and filter SIRENE database
- **api.collect**: External API data collection
- **api.apify**: Apify platform scrapers (Google Maps, LinkedIn)
- **http.static**: Web scraping
- **feeds.collect**: RSS/Atom feeds

### 2. Data Processing  
- **parse.html/pdf**: Extract structured data
- **normalize.standardize**: Clean and standardize data
- **enrich.address**: Address-based enrichment
- **enrich.google_maps**: Google Maps integration
- **enrich.domain/email/phone**: Contact discovery

### 3. Quality Control
- **quality.checks**: Data validation
- **quality.score**: Quality scoring
- **package.export**: Final export with reports

## Batch Processing

Process multiple business categories efficiently:

```bash
# Process multiple NAF codes
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --naf 6910Z \
  --input data/sirene.parquet \
  --output-dir out/professional_services \
  --profile standard \
  --continue-on-error
```

Creates organized output structure:
```
out/professional_services/
├── naf_6920Z/          # Accountants
├── naf_4329A/          # Construction  
├── naf_6910Z/          # Legal services
└── ...
```

## Budget and KPI System

Control resource usage and quality targets:

```yaml
# Resource budgets
budgets:
  max_http_requests: 2000    # HTTP request limit
  max_http_bytes: 52428800   # 50MB download limit
  time_budget_min: 90        # 90 minute time limit
  ram_mb: 4096              # 4GB RAM limit

# Quality targets  
kpi_targets:
  min_quality_score: 80          # Minimum quality score
  min_email_plausible_pct: 60    # Minimum email discovery rate
  min_domain_resolved_pct: 80    # Minimum domain validation rate
```

## Examples

### Professional Services Processing
```bash
# Accountants and legal services
python builder_cli.py batch \
  --naf 6920Z --naf 6910Z \
  --input data/sirene.parquet \
  --output-dir out/professional \
  --profile standard
```

### High-Volume Processing
```bash
# Large dataset with memory management
python builder_cli.py run-profile \
  --job jobs/large_dataset.yaml \
  --input data/sirene_full.parquet \
  --out out/large \
  --profile standard \
  --workers 4 \
  --max-ram-mb 8192
```

### Development Testing
```bash
# Quick development test
python builder_cli.py run-profile \
  --job jobs/test.yaml \
  --input data/sample.parquet \
  --out out/dev \
  --profile quick \
  --sample 20 \
  --debug
```

## Troubleshooting

### Common Issues

**Memory Issues**: Reduce `--workers`, set `--max-ram-mb`, use `--sample`
```bash
python builder_cli.py run-profile --job jobs/my_job.yaml --workers 2 --max-ram-mb 2048 --sample 100
```

**Network Issues**: Check connectivity, configure proxy in `.env`
```bash
# In .env file:
HTTP_PROXY=http://proxy:port
```

**API Rate Limits**: Reduce request rates in job configuration
```yaml
apify:
  google_places:
    max_places_per_search: 5  # Reduce from default 10
```

**No Results**: Verify input data format and required columns (`siren`, `naf_code`, `denomination`)

### Debug Mode

Enable comprehensive logging:
```bash
python builder_cli.py run-profile --job jobs/debug.yaml --input data.parquet --out out/debug --debug --verbose --sample 5
```

### Performance Optimization

- **Use SSD storage** for temporary files
- **Increase workers** (but watch memory usage): `--workers 8`
- **Optimize job configuration** (reduce unnecessary enrichment steps)
- **Use appropriate profile** (quick vs standard vs deep)

---

For detailed documentation on specific features, see the [Documentation](#documentation) section above.
