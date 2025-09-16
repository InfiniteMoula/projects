# Data Scraper and Enrichment Pipeline

A comprehensive data scraping and enrichment system designed to collect, process, and enrich business data from various sources including websites, APIs, and databases. The system is particularly optimized for processing French business data (SIRENE database) and enriching it with contact information, domain data, and quality scoring.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Processing Profiles](#processing-profiles)
- [Job Configuration](#job-configuration)
- [Pipeline Steps](#pipeline-steps)
- [Batch Processing](#batch-processing)
- [Input/Output Formats](#inputoutput-formats)
- [Budget and KPI System](#budget-and-kpi-system)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Features

- **Multi-source data collection**: Web scraping, API calls, RSS feeds, PDF extraction
- **Intelligent enrichment**: Email discovery, domain validation, phone normalization
- **Quality control**: Deduplication, scoring, validation checks
- **Flexible profiles**: Quick, standard, and deep processing modes
- **Budget management**: Control resource usage (time, RAM, HTTP requests)
- **Batch processing**: Process multiple NAF codes efficiently
- **Export formats**: CSV, Parquet, with quality reports (HTML/PDF)
- **Comprehensive logging**: Detailed execution tracking and debugging

## Installation

### Prerequisites

- Python 3.8+
- At least 2GB RAM (configurable)
- Internet connection for external data sources

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

The system requires these key dependencies:
- `pandas==2.2.2` - Data manipulation
- `httpx==0.27.0` - HTTP client
- `beautifulsoup4==4.12.3` - HTML parsing
- `PyYAML==6.0.2` - Configuration files
- `psutil==5.9.8` - System monitoring
- `weasyprint==66.0` - PDF generation
- `dnspython==2.6.1` - DNS resolution

## Environment Setup

### 1. Create Environment File

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# API Keys
HUNTER_API_KEY=your_hunter_api_key_here

# Proxy Settings (optional)
HTTP_PROXY=http://proxy:port
HTTPS_PROXY=https://proxy:port
```

### 3. Environment Variables Explained

- **HUNTER_API_KEY**: API key for Hunter.io email validation service (optional)
- **HTTP_PROXY/HTTPS_PROXY**: Proxy configuration for HTTP requests (optional)

## Quick Start

### 1. Prepare Input Data

Ensure you have a SIRENE dataset in Parquet or CSV format:

```bash
# Download or prepare your SIRENE data
# File should contain at least: siren, naf_code, denomination columns
```

### 2. Run a Quick Test

```bash
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_latest.parquet \
  --out out/test_run \
  --profile quick \
  --sample 50 \
  --dry-run
```

### 3. Full Processing

Remove `--dry-run` and `--sample` for full processing:

```bash
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_latest.parquet \
  --out out/experts_comptables \
  --profile standard
```

## Command Line Interface

### Main Commands

The CLI provides three main commands:

```bash
python builder_cli.py {run-step,run-profile,batch,resume}
```

### 1. run-profile

Execute a complete processing profile:

```bash
python builder_cli.py run-profile [OPTIONS]
```

**Required Arguments:**
- `--job JOB`: Path to job configuration file
- `--out OUT`: Output directory
- `--profile {quick,standard,deep}`: Processing profile

**Optional Arguments:**
- `--input INPUT`: Input data file (Parquet/CSV)
- `--run-id RUN_ID`: Custom run identifier
- `--dry-run`: Simulate execution without processing
- `--sample SAMPLE`: Limit processing to N records
- `--time-budget-min TIME_BUDGET`: Maximum execution time in minutes
- `--workers WORKERS`: Number of worker threads (default: 8)
- `--json`: Output results in JSON format
- `--resume`: Resume previous interrupted run
- `--verbose`: Enable detailed logging with all process details
- `--debug`: Enable debug mode with important debug information
- `--max-ram-mb MAX_RAM`: RAM budget in MB (0 = unlimited)
- `--explain`: Show execution plan without running

### 2. run-step

Execute a single pipeline step:

```bash
python builder_cli.py run-step --step STEP_NAME [OPTIONS]
```

Available steps: `dumps.collect`, `api.collect`, `http.static`, `parse.html`, `normalize.standardize`, `enrich.email`, `quality.score`, `package.export`, etc.

### 3. batch

Process multiple NAF codes in batch:

```bash
python builder_cli.py batch [OPTIONS]
```

**Required Arguments:**
- `--naf NAF_CODES`: NAF code(s) to process (can be repeated)
- `--input INPUT`: Input data file
- `--output-dir OUTPUT_DIR`: Base output directory

**Optional Arguments:**
- `--template TEMPLATE`: Job template file (default: job_template.yaml)
- `--profile {quick,standard,deep}`: Processing profile (default: quick)
- `--jobs-dir JOBS_DIR`: Generated job files directory (default: jobs_generated)
- `--dry-run`: Generate jobs without executing
- `--continue-on-error`: Continue processing if some jobs fail
- `--json`: Output results in JSON format
- `--verbose`: Enable detailed logging with all process details  
- `--debug`: Enable debug mode with important debug information

## Debug and Verbose Modes

The CLI provides two levels of enhanced logging to help with debugging and monitoring:

### Debug Mode (`--debug`)
Enables important debug information at crucial pipeline steps:
- Step start/completion notifications with status
- Pipeline configuration overview 
- Budget and KPI status tracking
- RAM usage monitoring
- Error summaries and step results
- Final pipeline statistics

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --input data/sirene.parquet \
  --out out/debug_run \
  --profile quick \
  --debug
```

### Verbose Mode (`--verbose`) 
Enables comprehensive detailed logging of all process details:
- All debug mode information plus:
- Detailed step configurations and context
- Complete step output and results
- Full job configuration details  
- Complete KPI and budget breakdowns
- Full pipeline execution summary
- Enhanced log formatting with file/line numbers

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --input data/sirene.parquet \
  --out out/verbose_run \
  --profile standard \
  --verbose
```

### Debug Mode Usage Examples

**Basic debug run:**
```bash
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene.parquet \
  --out out/debug_test \
  --profile quick \
  --debug \
  --sample 10
```

**Verbose batch processing:**
```bash
python builder_cli.py batch \
  --naf 6920Z --naf 4329A \
  --input data/sirene.parquet \
  --output-dir out/verbose_batch \
  --verbose \
  --sample 50
```

**Debug mode with budget constraints:**
```bash
python builder_cli.py run-profile \
  --job jobs/btp_idf.yaml \
  --input data/sirene.parquet \
  --out out/debug_budget \
  --profile standard \
  --debug \
  --max-ram-mb 1024 \
  --time-budget-min 15
```

### Log Output Examples

**Debug mode output:**
```
2024-01-15T10:30:00 [INFO] builder.pipeline:532 - [DEBUG] Pipeline configuration:
2024-01-15T10:30:00 [INFO] builder.pipeline:533 - [DEBUG] - Profile: quick
2024-01-15T10:30:00 [INFO] builder.pipeline:534 - [DEBUG] - Total steps: 7
2024-01-15T10:30:00 [INFO] builder.pipeline:535 - [DEBUG] - Steps: dumps.collect, api.collect, normalize.standardize, quality.checks, quality.dedupe, quality.score, package.export
2024-01-15T10:30:01 [INFO] builder.pipeline:143 - [DEBUG] Starting step 'dumps.collect'
2024-01-15T10:30:01 [INFO] builder.pipeline:148 - [DEBUG] Resolved step function: dumps.collect_dump.run
```

**Verbose mode output:**
```
2024-01-15T10:30:00 [DEBUG] builder.pipeline:538 - [VERBOSE] Job configuration: {
  "niche": "experts_comptables",
  "profile": "quick",
  "filters": {"naf_include": ["6920Z"]},
  ...
}
2024-01-15T10:30:01 [DEBUG] builder.pipeline:147 - [VERBOSE] Step configuration keys: ['niche', 'filters', 'profile']
2024-01-15T10:30:01 [DEBUG] builder.pipeline:148 - [VERBOSE] Context keys: ['run_id', 'outdir', 'logs', 'dry_run']
```

### 4. resume

Resume an interrupted run:

```bash
python builder_cli.py resume [OPTIONS]
```

## Processing Profiles

The system offers three predefined processing profiles with different levels of depth and resource usage:

### Quick Profile
**Use case**: Fast processing, basic enrichment
**Steps**: Data collection → Normalization → Basic quality checks → Export
**Time**: ~5-15 minutes
**Resources**: Low

```yaml
steps: [
  "dumps.collect",
  "api.collect", 
  "normalize.standardize",
  "quality.checks",
  "quality.dedupe",
  "quality.score",
  "package.export"
]
```

### Standard Profile  
**Use case**: Balanced processing with web scraping
**Steps**: Quick + Web scraping + Domain/Email enrichment
**Time**: ~20-60 minutes
**Resources**: Medium

```yaml
steps: [
  "dumps.collect",
  "api.collect",
  "http.static",
  "http.sitemap", 
  "parse.jsonld",
  "normalize.standardize",
  "enrich.domain",
  "enrich.site",
  "enrich.dns",
  "enrich.email",
  "enrich.phone",
  "quality.checks",
  "quality.dedupe", 
  "quality.score",
  "package.export"
]
```

### Deep Profile
**Use case**: Comprehensive processing with PDF/HTML extraction
**Steps**: Standard + Headless browsing + PDF processing + HTML parsing
**Time**: ~1-3 hours
**Resources**: High

```yaml
steps: [
  "dumps.collect",
  "api.collect",
  "http.static",
  "http.sitemap",
  "headless.collect",
  "pdf.collect",
  "parse.pdf",
  "parse.html", 
  "parse.jsonld",
  "normalize.standardize",
  "enrich.domain",
  "enrich.site",
  "enrich.dns",
  "enrich.email",
  "enrich.phone",
  "quality.checks",
  "quality.dedupe",
  "quality.score", 
  "package.export"
]
```

## Job Configuration

Jobs are configured using YAML files that define the processing parameters, data sources, and quality targets.

### Basic Job Structure

```yaml
niche: "job_name"

# Data filtering
filters:
  naf_include: ["6920Z"]  # NAF codes to include
  active_only: false      # Only active businesses
  regions: ["75", "92"]   # Postal code prefixes (optional)

# Processing profile
profile: "standard"

# Web scraping configuration
http:
  seeds: ["https://example.com"]  # Starting URLs
  per_domain_rps: 0.5            # Requests per second limit

# Sitemap processing
sitemap:
  domains: ["example.com"]
  allow_patterns: ["contact", "about"]
  max_urls: 500

# Data enrichment
enrich:
  directory_csv: ""  # External directory file
  email_formats_priority: ["contact@{d}", "info@{d}"]

# Quality configuration
dedupe:
  keys: ["siren", "domain_root", "best_email"]
  fuzzy: false

scoring:
  weights: {contactability:50, unicity:20, completeness:20, freshness:10}

# Output configuration
output:
  dir: "out/job_name"
  lang: "fr"

# Quality targets (KPIs)
kpi_targets:
  min_quality_score: 80
  max_dup_pct: 1.5
  min_url_valid_pct: 85
  min_domain_resolved_pct: 80
  min_email_plausible_pct: 60
  min_lines_per_s: 50

# Resource budgets
budgets:
  max_http_bytes: 10485760    # 10MB
  max_http_requests: 500      # Maximum HTTP requests
  time_budget_min: 30         # 30 minutes
  ram_mb: 2048               # 2GB RAM limit

retention_days: 30
```

### Creating Custom Jobs

1. **Copy the template**:
   ```bash
   cp job_template.yaml jobs/my_custom_job.yaml
   ```

2. **Edit the configuration**:
   - Update `niche` name
   - Set appropriate `naf_include` codes
   - Configure `http.seeds` for web scraping
   - Adjust `budgets` for your resources
   - Set `kpi_targets` for quality requirements

3. **Test with dry run**:
   ```bash
   python builder_cli.py run-profile \
     --job jobs/my_custom_job.yaml \
     --input data/sirene.parquet \
     --out out/test \
     --profile quick \
     --dry-run \
     --sample 10
   ```

## Pipeline Steps

The system is built as a modular pipeline with discrete steps. Each step can be run independently or as part of a profile.

### Data Collection Steps

#### 1. dumps.collect
**Purpose**: Load and filter input data (SIRENE database)
**Input**: Raw SIRENE Parquet/CSV file
**Output**: Filtered dataset based on NAF codes and regions
**Configuration**: `filters.naf_include`, `filters.regions`, `filters.active_only`

#### 2. api.collect  
**Purpose**: Collect data from external APIs
**Input**: Filtered dataset
**Output**: API-enriched data
**Configuration**: `api.endpoints`

#### 3. http.static
**Purpose**: Scrape static web pages
**Input**: URLs from dataset or seeds
**Output**: Downloaded HTML content
**Configuration**: `http.seeds`, `http.per_domain_rps`

#### 4. http.sitemap
**Purpose**: Discover URLs from sitemaps
**Input**: Domain list
**Output**: Discovered URLs
**Configuration**: `sitemap.domains`, `sitemap.allow_patterns`, `sitemap.max_urls`

#### 5. headless.collect
**Purpose**: Scrape dynamic content using headless browser
**Input**: URLs requiring JavaScript
**Output**: Rendered HTML content
**Dependencies**: Requires `http.static`

#### 6. feeds.collect
**Purpose**: Collect data from RSS/Atom feeds
**Input**: Feed URLs
**Output**: Feed content and metadata
**Configuration**: `feeds.urls`

#### 7. pdf.collect
**Purpose**: Download and collect PDF documents
**Input**: PDF URLs
**Output**: PDF files
**Configuration**: `pdf.urls`

### Data Parsing Steps

#### 8. parse.html
**Purpose**: Extract structured data from HTML content
**Input**: HTML files from web scraping
**Output**: Structured contact information
**Dependencies**: `http.static`, `headless.collect`

#### 9. parse.jsonld
**Purpose**: Extract JSON-LD structured data
**Input**: HTML with JSON-LD markup
**Output**: Structured metadata
**Dependencies**: `http.static`, `http.sitemap`

#### 10. parse.pdf
**Purpose**: Extract text and data from PDF documents
**Input**: PDF files
**Output**: Extracted text and metadata
**Dependencies**: `pdf.collect`

### Data Processing Steps

#### 11. normalize.standardize
**Purpose**: Standardize and clean collected data
**Input**: Raw collected data from multiple sources
**Output**: Standardized, consistent dataset
**Processing**: Phone normalization, email validation, address parsing

### Data Enrichment Steps

#### 12. enrich.domain
**Purpose**: Discover and validate domains for businesses
**Input**: Standardized dataset
**Output**: Domain information per business
**Dependencies**: `normalize.standardize`

#### 13. enrich.site
**Purpose**: Probe websites for additional information
**Input**: Domain list
**Output**: Website metadata, technology stack
**Dependencies**: `enrich.domain`

#### 14. enrich.dns
**Purpose**: Perform DNS validation and checks
**Input**: Domain list
**Output**: DNS validation results
**Dependencies**: `enrich.domain`

#### 15. enrich.email
**Purpose**: Discover and validate email addresses
**Input**: Domains and contact patterns
**Output**: Validated email addresses
**Dependencies**: `enrich.dns`
**Configuration**: `enrich.email_formats_priority`

#### 16. enrich.phone
**Purpose**: Enrich and validate phone numbers
**Input**: Phone data from various sources
**Output**: Validated, normalized phone numbers
**Dependencies**: `enrich.email`

### Quality Control Steps

#### 17. quality.checks
**Purpose**: Perform data quality validation
**Input**: Enriched dataset
**Output**: Quality metrics and validation results
**Dependencies**: `normalize.standardize`

#### 18. quality.dedupe
**Purpose**: Remove duplicate records
**Input**: Quality-checked dataset
**Output**: Deduplicated dataset
**Configuration**: `dedupe.keys`, `dedupe.fuzzy`
**Dependencies**: `enrich.email`, `normalize.standardize`

#### 19. quality.score
**Purpose**: Calculate quality scores for each record
**Input**: Deduplicated dataset
**Output**: Dataset with quality scores
**Configuration**: `scoring.weights`
**Dependencies**: `quality.dedupe`

### Export Step

#### 20. package.export
**Purpose**: Generate final outputs and reports
**Input**: Scored dataset
**Output**: CSV, Parquet, HTML/PDF reports, metadata
**Dependencies**: `quality.score`
**Files generated**:
  - `dataset.csv` - Final CSV export
  - `dataset.parquet` - Final Parquet export
  - `data_quality_report.html` - Interactive quality report
  - `data_quality_report.pdf` - PDF quality report
  - `manifest.json` - Run metadata and quality metrics
  - `sha256.txt` - File checksums

## Batch Processing

The batch processing feature allows you to process multiple NAF codes efficiently in a single operation.

### Basic Batch Usage

```bash
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --naf 43 \
  --input data/sirene_latest.parquet \
  --output-dir out/batch_results
```

### Batch Options

- **Multiple NAF codes**: Use `--naf` multiple times
- **Custom template**: `--template custom_template.yaml`
- **Processing profile**: `--profile {quick,standard,deep}`
- **Dry run**: `--dry-run` to generate jobs without executing
- **Error handling**: `--continue-on-error` to process all codes even if some fail
- **Resource limits**: `--sample`, `--workers`, `--max-ram-mb`

### Generated Structure

Batch processing creates organized output:

```
output-dir/
├── naf_6920Z/           # Job output for NAF 6920Z
│   ├── dataset.csv
│   ├── dataset.parquet
│   ├── data_quality_report.html
│   └── logs/
├── naf_4329A/           # Job output for NAF 4329A
│   ├── dataset.csv
│   ├── dataset.parquet
│   └── logs/
└── ...

jobs_generated/          # Generated job files
├── naf_6920Z.yaml
├── naf_4329A.yaml
└── ...
```

### Advanced Batch Examples

**Generate jobs only (dry run)**:
```bash
python builder_cli.py batch \
  --naf 6920Z --naf 4329A \
  --input data/sirene.parquet \
  --output-dir out/batch \
  --dry-run \
  --verbose
```

**With error tolerance**:
```bash
python builder_cli.py batch \
  --naf 6920Z --naf 4329A --naf 43 \
  --input data/sirene.parquet \
  --output-dir out/batch \
  --continue-on-error \
  --verbose
```

**JSON output for automation**:
```bash
python builder_cli.py batch \
  --naf 6920Z --naf 4329A \
  --input data/sirene.parquet \
  --output-dir out/batch \
  --json > batch_results.json
```

## Input/Output Formats

### Input Data Requirements

The system expects input data in Parquet or CSV format with these required columns:

**Minimum required columns**:
- `siren`: Business identifier (SIREN number)
- `naf_code`: NAF activity code
- `denomination`: Business name

**Optional but recommended columns**:
- `adresse`: Business address
- `code_postal`: Postal code
- `ville`: City
- `telephone`: Phone number
- `site_web`: Website URL
- `email`: Email address
- `effectif`: Employee count
- `date_creation`: Creation date

### Output Data Structure

The final dataset contains enriched data with these key columns:

**Business Information**:
- `siren`, `denomination`, `naf_code`
- `adresse_complete`, `code_postal`, `ville`
- `effectif`, `date_creation`

**Contact Information**:
- `telephone_norm`: Normalized phone number
- `best_email`: Best discovered email address
- `site_web`: Website URL
- `domain_root`: Root domain

**Enrichment Data**:
- `domain_valid`: Domain validation status
- `dns_resolved`: DNS resolution status
- `email_plausible`: Email plausibility score
- `contact_score`: Contact information quality

**Quality Metrics**:
- `quality_score`: Overall quality score (0-100)
- `completeness`: Data completeness percentage
- `contactability`: Contact information availability
- `freshness`: Data recency score

### Export Files

Each successful run generates these files:

1. **dataset.csv**: Final enriched dataset in CSV format
2. **dataset.parquet**: Final enriched dataset in Parquet format  
3. **data_quality_report.html**: Interactive quality report
4. **data_quality_report.pdf**: PDF version of quality report
5. **manifest.json**: Run metadata and quality metrics
6. **sha256.txt**: File integrity checksums
7. **logs/run_id.jsonl**: Detailed execution logs

## Budget and KPI System

The system includes comprehensive budget management and quality KPI tracking.

### Budget Configuration

Control resource usage with budget limits:

```yaml
budgets:
  max_http_bytes: 10485760    # Maximum HTTP download (10MB)
  max_http_requests: 500      # Maximum HTTP requests
  time_budget_min: 30         # Maximum execution time (30 min)
  ram_mb: 2048               # Maximum RAM usage (2GB)
```

### KPI Targets

Define quality targets for automatic validation:

```yaml
kpi_targets:
  min_quality_score: 80          # Minimum average quality score
  max_dup_pct: 1.5              # Maximum duplicate percentage  
  min_url_valid_pct: 85         # Minimum valid URL percentage
  min_domain_resolved_pct: 80   # Minimum DNS resolution rate
  min_email_plausible_pct: 60   # Minimum plausible email rate
  min_lines_per_s: 50           # Minimum processing speed
```

### Budget Monitoring

Monitor resource usage during execution:

- **HTTP Budget**: Tracks download size and request count
- **Time Budget**: Monitors execution time vs. limits
- **RAM Budget**: Monitors memory usage

Budget exceeded scenarios will:
1. Log warnings when approaching limits
2. Gracefully stop processing when limits exceeded
3. Include budget statistics in final report

### KPI Validation

After processing, KPIs are automatically calculated and validated:

- **Quality Score**: Average quality across all records
- **Duplicate Rate**: Percentage of duplicate records found
- **URL Validation**: Percentage of valid/accessible URLs
- **Domain Resolution**: Percentage of domains that resolve via DNS
- **Email Plausibility**: Percentage of emails passing validation
- **Processing Speed**: Records processed per second

KPI failures will be reported in the logs and final report.

## Examples

### Example 1: Quick Expert-Comptable Processing

Process expert-comptable businesses with minimal resources:

```bash
python builder_cli.py run-profile \
  --job jobs/experts_comptables.yaml \
  --input data/sirene_latest.parquet \
  --out out/experts_comptables \
  --profile quick \
  --sample 100 \
  --verbose
```

### Example 2: Deep BTP Processing

Comprehensive processing for construction businesses:

```bash
python builder_cli.py run-profile \
  --job jobs/btp_idf.yaml \
  --input data/sirene_idf.parquet \
  --out out/btp_deep \
  --profile deep \
  --workers 4 \
  --max-ram-mb 4096
```

### Example 3: Batch NAF Processing

Process multiple NAF codes with error tolerance:

```bash
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --naf 4711F \
  --naf 43 \
  --input data/sirene_latest.parquet \
  --output-dir out/multi_naf \
  --profile standard \
  --continue-on-error \
  --verbose
```

### Example 4: Custom Job Creation

Create and run a custom job for a specific niche:

```bash
# 1. Create job from template
python create_job.py jobs/my_niche.yaml

# 2. Edit the generated file
# Edit naf_include, http.seeds, budgets, etc.

# 3. Test with dry run
python builder_cli.py run-profile \
  --job jobs/my_niche.yaml \
  --input data/sirene.parquet \
  --out out/test \
  --profile quick \
  --dry-run \
  --sample 10

# 4. Full processing
python builder_cli.py run-profile \
  --job jobs/my_niche.yaml \
  --input data/sirene.parquet \
  --out out/my_niche \
  --profile standard
```

### Example 5: Resume Interrupted Processing

Resume a processing run that was interrupted:

```bash
python builder_cli.py resume \
  --job jobs/experts_comptables.yaml \
  --out out/experts_comptables \
  --run-id abc123def456
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: Process killed due to high memory usage

**Solutions**:
- Reduce `--workers` (default: 8)
- Set `--max-ram-mb` to limit memory
- Use `--sample` for testing with smaller datasets
- Choose `quick` profile instead of `deep`

```bash
# Memory-optimized run
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --input data/sirene.parquet \
  --out out/test \
  --profile quick \
  --workers 2 \
  --max-ram-mb 1024
```

#### 2. Network/HTTP Issues
**Problem**: HTTP requests failing or timing out

**Solutions**:
- Check internet connectivity
- Configure proxy settings in `.env`
- Reduce `http.per_domain_rps` in job config
- Check if target domains are blocking requests

#### 3. Budget Exceeded Errors
**Problem**: Processing stops due to budget limits

**Solutions**:
- Increase budget limits in job configuration
- Use `--time-budget-min 0` to disable time limits  
- Reduce data size with `--sample`
- Optimize job configuration

#### 4. Input Data Issues
**Problem**: Missing required columns or data format errors

**Solutions**:
- Verify input file has required columns: `siren`, `naf_code`, `denomination`
- Check file format (CSV/Parquet)
- Validate data encoding (UTF-8)

```bash
# Check input file structure
python -c "import pandas as pd; print(pd.read_parquet('data/sirene.parquet').columns.tolist())"
```

#### 5. Job Configuration Errors
**Problem**: YAML syntax errors or invalid configuration

**Solutions**:
- Validate YAML syntax
- Check against `job_template.yaml`
- Use `--explain` to see execution plan
- Start with `--dry-run`

### Debug Mode

Enable maximum verbosity for debugging:

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --input data/sirene.parquet \
  --out out/debug \
  --profile quick \
  --debug \
  --sample 5
```

### Verbose Mode

Enable comprehensive detailed logging:

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --input data/sirene.parquet \
  --out out/verbose \
  --profile quick \
  --verbose \
  --sample 5
```

### Log Analysis

Check execution logs for detailed information:

```bash
# View latest log file
ls -la out/my_run/logs/
tail -f out/my_run/logs/run_123abc.jsonl
```

### Performance Optimization

**For faster processing**:
- Use `quick` profile for basic processing
- Increase `--workers` (but watch memory usage)
- Use SSD storage for temp files
- Ensure sufficient RAM

**For memory optimization**:
- Reduce `--workers`
- Set `--max-ram-mb` limit
- Process in smaller batches
- Use `--sample` for testing

## Development

### Project Structure

```
├── builder_cli.py          # Main CLI interface
├── create_job.py           # Job generation utilities
├── job_template.yaml       # Default job template
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── jobs/                  # Job configuration files
├── utils/                 # Core utilities
│   ├── pipeline.py        # Pipeline orchestration
│   ├── config.py          # Configuration management
│   ├── budget_middleware.py # Budget and KPI tracking
│   └── ...
├── dumps/                 # Data collection modules
├── parse/                 # Data parsing modules  
├── enrich/                # Data enrichment modules
├── quality/               # Quality control modules
├── package/               # Export and reporting
└── tests/                 # Test suite
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_builder_cli.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Contributing

1. **Code Style**: Follow PEP 8 standards
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Logging**: Use the pipeline logging system

### Adding New Pipeline Steps

1. Create module in appropriate directory
2. Implement `run(config, context)` function
3. Add to `STEP_REGISTRY` in `builder_cli.py`
4. Define dependencies in `STEP_DEPENDENCIES`
5. Add tests in `tests/`

### Custom Enrichment Modules

```python
# Example: enrich/custom_enrichment.py
def run(config, context):
    """Custom enrichment step"""
    logger = context.get("logger")
    input_path = context["outdir_path"] / "previous_step.parquet"
    
    # Process data
    df = pd.read_parquet(input_path)
    # ... enrichment logic ...
    
    # Save results
    output_path = context["outdir_path"] / "custom_enriched.parquet"
    df.to_parquet(output_path)
    
    return {"status": "OK", "records_processed": len(df)}
```

---

For more detailed information, see the specific documentation files:
- [BATCH_GUIDE.md](BATCH_GUIDE.md) - Batch processing documentation
- [BUDGET_MIDDLEWARE.md](BUDGET_MIDDLEWARE.md) - Budget and KPI system details
- [niche_guide.md](niche_guide.md) - Niche development guide
- [package/README.md](package/README.md) - Export system documentation

For support or questions, please check the logs first, then refer to the troubleshooting section above.