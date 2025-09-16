# Budget Tracking & KPI Monitoring Middleware

This document describes the budget tracking and KPI monitoring middleware implemented for the pipeline system.

## Overview

The middleware provides comprehensive resource usage tracking and KPI monitoring with automatic enforcement of configured limits. It seamlessly integrates into the existing pipeline system without breaking changes.

## Features

### 🛡️ Budget Tracking & Enforcement

- **HTTP Request Counting**: Tracks and limits the number of HTTP requests made during pipeline execution
- **HTTP Bytes Tracking**: Monitors total response bytes downloaded with configurable limits  
- **Time Budget Management**: Enforces maximum execution time limits per pipeline run
- **RAM Usage Monitoring**: Integration with existing RAM budget checking
- **Graceful Failure**: Pipeline stops cleanly when budgets are exceeded with clear error messages

### 📊 KPI Calculation & Monitoring

- **Quality Score Tracking**: Monitors data quality scores against minimum thresholds
- **Duplicate Rate Monitoring**: Tracks and enforces maximum duplicate percentage limits
- **URL Validation Rates**: Monitors percentage of valid URLs processed
- **Domain Resolution Rates**: Tracks DNS resolution success rates
- **Email Plausibility Rates**: Monitors email validation success rates  
- **Processing Throughput**: Calculates and monitors lines processed per second

### 🔧 Integration Features

- **Context Integration**: Budget tracker available to all pipeline steps via context
- **HTTP Middleware**: Automatic request tracking in HTTP utilities
- **Comprehensive Logging**: Per-step budget usage and KPI status logging
- **JSON Output Enhancement**: Budget and KPI data included in pipeline results
- **Backward Compatibility**: Works with existing job configurations

## Configuration

### Budget Configuration (in job YAML)

```yaml
budgets:
  max_http_requests: 300     # Maximum HTTP requests allowed
  max_http_bytes: 10485760   # Maximum bytes to download (10MB)
  time_budget_min: 20        # Maximum execution time in minutes
  ram_mb: 2048              # Maximum RAM usage in MB
```

### KPI Targets Configuration (in job YAML)

```yaml
kpi_targets:
  min_quality_score: 80         # Minimum quality score required
  max_dup_pct: 1.5             # Maximum duplicate percentage allowed
  min_url_valid_pct: 85        # Minimum URL validation rate required
  min_domain_resolved_pct: 80  # Minimum domain resolution rate required
  min_email_plausible_pct: 60  # Minimum email plausibility rate required
  min_lines_per_s: 50          # Minimum processing throughput required
```

## Usage Examples

### Basic Pipeline Run with Budget Tracking

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --out output/ \
  --profile standard \
  --verbose
```

### JSON Output with Budget and KPI Data

```bash
python builder_cli.py run-profile \
  --job jobs/my_job.yaml \
  --out output/ \
  --profile quick \
  --json
```

Example JSON output:
```json
{
  "run_id": "abc123def456",
  "results": [...],
  "outdir": "/path/to/output",
  "budget_stats": {
    "http_requests": 45,
    "max_http_requests": 300,
    "http_bytes": 2501234,
    "max_http_bytes": 10485760,
    "elapsed_min": 12.5,
    "time_budget_min": 20,
    "http_requests_pct": 15.0,
    "http_bytes_pct": 23.9,
    "time_budget_pct": 62.5
  },
  "kpis": {
    "actual_kpis": {
      "quality_score": 85.5,
      "dup_pct": 1.2,
      "url_valid_pct": 87.3,
      "domain_resolved_pct": 82.1,
      "email_plausible_pct": 65.8,
      "lines_per_s": 55.2
    },
    "all_kpis_met": true
  }
}
```

## Implementation Details

### Core Components

1. **`BudgetTracker`**: Tracks resource usage and enforces limits
2. **`KPICalculator`**: Extracts KPIs from pipeline results and compares against targets  
3. **`http_request_tracking`**: Context manager for automatic HTTP request tracking
4. **Pipeline Integration**: Modified `_run_step()` function with budget/KPI monitoring

### Key Files Modified

- **`utils/budget_middleware.py`** - New middleware implementation
- **`builder_cli.py`** - Pipeline integration and budget/KPI reporting
- **`utils/http.py`** - HTTP request tracking support
- **`tests/test_budget_middleware.py`** - Comprehensive test suite

### Error Handling

When budgets are exceeded, the middleware raises `BudgetExceededError` with clear messages:

- `"HTTP request budget exceeded: 305 > 300"`
- `"HTTP bytes budget exceeded: 10500000 > 10485760"`  
- `"Time budget exceeded: 21.5 min > 20 min"`

## Testing

The implementation includes comprehensive tests covering:

- Budget tracker creation and configuration
- HTTP request and bytes limit enforcement
- Time budget enforcement  
- KPI calculation accuracy
- Integration with pipeline system

Run tests with:
```bash
python -m pytest tests/test_budget_middleware.py -v
```

## Benefits

1. **Resource Control**: Prevents runaway processes that exceed allocated resources
2. **Quality Assurance**: Ensures pipeline outputs meet defined quality standards
3. **Cost Management**: Controls HTTP usage and processing time to manage costs
4. **Monitoring**: Provides detailed visibility into resource usage and quality metrics
5. **Automation**: Automatic enforcement without manual intervention
6. **Flexibility**: Configurable limits per job profile or environment

## Example Job Configurations

See `jobs/budget_test.yaml` and `jobs/strict_budget_test.yaml` for example configurations demonstrating the middleware functionality.