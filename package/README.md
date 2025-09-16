# Enhanced Package Export Step

## Overview

The enhanced `package.export` step automates the final CSV export with comprehensive data quality reporting. It assembles `quality_score.parquet`, `deduped.parquet` and metadata to deliver:

- **Final formatted CSV** with quality scores merged
- **HTML quality report** with interactive metrics
- **PDF report** generated from HTML
- **Enhanced metadata** including quality metrics

## Features

### 1. Data Merging
- Merges `quality_score.parquet` with `deduped.parquet` on index
- Falls back to other sources if deduped data is missing:
  - `enriched_email.parquet`
  - `enriched_domain.parquet` 
  - `normalized.parquet`
- Handles missing quality scores gracefully

### 2. Quality Metrics
Calculates comprehensive quality metrics:
- **Total records**
- **Quality score mean** (percentage)
- **Quality score median (P50)**
- **Quality score P90**

### 3. HTML Report
Generates an interactive HTML report (`data_quality_report.html`) including:
- Executive summary with key metrics
- Quality score distribution visualization
- Data dictionary with completeness rates
- Technical metadata (file paths, checksums)
- Compliance and governance information

### 4. PDF Report
Automatically generates PDF version (`data_quality_report.pdf`) using WeasyPrint.

### 5. Enhanced Outputs

#### Files Created:
- `dataset.csv` - Final CSV with quality scores
- `dataset.parquet` - Final Parquet file
- `data_quality_report.html` - Interactive HTML report
- `data_quality_report.pdf` - PDF report
- `manifest.json` - Enhanced metadata with quality metrics
- `data_dictionary.md` - Data dictionary (backward compatibility)
- `sha256.txt` - Checksums for all files

#### Enhanced Manifest Structure:
```json
{
  "run_id": "run_123",
  "dataset_id": "dataset_hash",
  "records": 1000,
  "paths": {
    "csv": "/path/to/dataset.csv",
    "parquet": "/path/to/dataset.parquet", 
    "html_report": "/path/to/data_quality_report.html",
    "pdf_report": "/path/to/data_quality_report.pdf"
  },
  "quality_metrics": {
    "total_records": 1000,
    "quality_mean": 85.2,
    "quality_p50": 87.5,
    "quality_p90": 95.1
  },
  "manifest": {
    "robots_compliance": true,
    "tos_breaches": [],
    "pii_present": false,
    "anonymization_used": false
  }
}
```

## Pipeline Integration

The step integrates seamlessly with the existing pipeline:

- **Registry**: `package.exporter:run`
- **Dependencies**: `quality.score`
- **Profiles**: Included in `quick`, `standard`, and `deep` profiles

## Error Handling

The enhanced exporter handles various error conditions gracefully:

- **Missing quality scores**: Reports generated without quality metrics
- **Missing source data**: Returns `FAIL` status with error message
- **PDF generation failure**: Continues with HTML report, logs warning
- **Template errors**: Falls back to basic reporting

## Backward Compatibility

The enhanced exporter maintains full backward compatibility:
- Existing `data_dictionary.md` format preserved
- `sha256.txt` checksums include all files
- Manifest structure extends existing format
- CSV/Parquet outputs remain unchanged

## Usage Example

```python
# Via pipeline
python builder_cli.py run-profile --job jobs/example.yaml --out ./output --profile quick

# Direct usage
from package import exporter

ctx = {
    "run_id": "test_run",
    "outdir_path": Path("./output"),
    "lang": "fr"
}

result = exporter.run({}, ctx)
```

## Dependencies

Additional dependencies required:
- `jinja2==3.1.4` - HTML template rendering
- `weasyprint==66.0` - PDF generation

## Testing

Comprehensive test suite covers:
- Data merging functionality
- Quality metrics calculation  
- Report generation
- Error handling scenarios
- Pipeline integration

Run tests: `python -m pytest tests/test_enhanced_exporter.py -v`