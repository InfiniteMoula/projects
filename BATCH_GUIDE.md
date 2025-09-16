# Batch Job Generator Documentation

This document describes the new batch job generation functionality that allows processing multiple NAF codes in an industrialized way.

## Overview

The batch job generator automates the creation and execution of multiple jobs from a list of NAF codes. It provides:

1. **Dynamic YAML generation**: Creates job configuration files from templates
2. **Sequential execution**: Runs jobs one after another with proper error handling
3. **Flexible configuration**: Supports different profiles and custom parameters

## Commands

### Batch Processing

```bash
python builder_cli.py batch --naf 6920Z --naf 4329A --input data/sirene_latest.parquet --output-dir out/batch_results
```

#### Required Arguments

- `--naf NAF_CODES`: NAF code(s) to process (can be used multiple times)
- `--input INPUT`: Input file for processing
- `--output-dir OUTPUT_DIR`: Base output directory for all jobs

#### Optional Arguments

- `--template TEMPLATE`: Path to job template file (default: job_template.yaml)
- `--profile {quick,standard,deep}`: Profile to use for jobs (default: quick)
- `--jobs-dir JOBS_DIR`: Directory to store generated job files (default: jobs_generated)
- `--dry-run`: Generate jobs but don't run them
- `--sample SAMPLE`: Sample size for testing
- `--workers WORKERS`: Number of workers
- `--verbose`: Verbose output
- `--max-ram-mb MAX_RAM_MB`: Maximum RAM budget in MB
- `--continue-on-error`: Continue processing other NAF codes if one fails
- `--json`: Output results in JSON format

### Standalone Job Generation

```bash
python create_job.py /path/to/output --naf 6920Z --batch
```

## Examples

### Basic Batch Processing
```bash
# Process two NAF codes with quick profile
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --input data/sirene_latest.parquet \
  --output-dir out/batch_results \
  --profile quick
```

### Dry Run (Generate Only)
```bash
# Generate job files without executing them
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --input data/sirene_latest.parquet \
  --output-dir out/batch_results \
  --dry-run \
  --verbose
```

### With Error Tolerance
```bash
# Continue processing even if some jobs fail
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --naf 43 \
  --input data/sirene_latest.parquet \
  --output-dir out/batch_results \
  --continue-on-error \
  --verbose
```

### JSON Output
```bash
# Get results in JSON format
python builder_cli.py batch \
  --naf 6920Z \
  --naf 4329A \
  --input data/sirene_latest.parquet \
  --output-dir out/batch_results \
  --json
```

## Job Template

The default job template (`job_template.yaml`) provides a flexible base configuration that can be customized. Key template variables:

- `{niche_name}`: Generated from NAF code (e.g., "naf_6920Z")
- `{naf_code}`: The original NAF code
- `{profile}`: Processing profile (quick/standard/deep)

### Custom Templates

You can create custom templates and use them with `--template`:

```yaml
niche: "{niche_name}"
filters:
  naf_include: ["{naf_code}"]
  active_only: true
profile: "{profile}"
# ... rest of configuration
```

## Directory Structure

When running batch jobs, the following structure is created:

```
output-dir/
├── naf_6920Z/           # Job output for NAF 6920Z
│   ├── dataset.csv
│   ├── dataset.parquet
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

## Error Handling

- **Fail Fast**: By default, batch processing stops on first error
- **Continue on Error**: Use `--continue-on-error` to process all NAF codes
- **Timeout Protection**: Each job has a 1-hour timeout
- **Detailed Logging**: Use `--verbose` for detailed progress information

## Return Codes

- `0`: Success (all jobs completed successfully)
- `1`: Error (at least one job failed, unless `--continue-on-error` is used)

## Integration with Existing Workflow

The batch functionality is designed to integrate seamlessly with the existing workflow:

1. **Generate jobs**: `python builder_cli.py batch --dry-run ...`
2. **Review generated configs**: Check files in `jobs_generated/`
3. **Execute batch**: Remove `--dry-run` flag
4. **Monitor progress**: Use `--verbose` for detailed logging
5. **Collect results**: Find outputs in individual NAF directories