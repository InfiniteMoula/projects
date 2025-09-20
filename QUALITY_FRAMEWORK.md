# Quality Control Framework

This directory contains the Quality Control Framework implementation for Week 3-4 of the Apify automation roadmap.

## Overview

The Quality Control Framework provides comprehensive validation, confidence scoring, and automated filtering for business data enrichment workflows. It includes specialized controllers for different data sources and integrates seamlessly with the existing Apify pipeline.

## Core Components

### 1. Contact Extractor (`utils/contact_extractor.py`)

Advanced contact information extraction and validation with confidence scoring.

**Features:**
- French phone number extraction and validation
- Email extraction with domain-based confidence scoring
- Website URL extraction and normalization
- Deep validation with DNS and HTTP accessibility checks
- Source-specific confidence adjustments

**Usage:**
```python
from utils.contact_extractor import ContactExtractor

extractor = ContactExtractor()
contact_info = extractor.extract_contact_info(text, source="google_maps")
print(f"Phone: {contact_info.phone}, Confidence: {contact_info.confidence_score}")
```

### 2. Quality Controller (`utils/quality_controller.py`)

Main quality control framework with specialized controllers for different data sources.

**Features:**
- Comprehensive validation framework for extraction results
- Specialized controllers for Google Maps and LinkedIn data
- Automated quality scoring and confidence levels
- Quality report generation with dashboard data export
- Automated filtering based on quality thresholds

**Usage:**
```python
from utils.quality_controller import GoogleMapsQualityController

controller = GoogleMapsQualityController()
validated_results = controller.validate_extraction_results(results, 'google_maps')
report = controller.generate_quality_report(validated_results)
```

### 3. Enhanced Apify Integration (`api/apify_agents.py`)

Quality validation integrated into the Apify workflow.

**Features:**
- Quality validation integrated into Apify workflow
- Automatic quality scoring for all results
- Quality dashboard JSON export for monitoring
- Configurable quality-based filtering

## Quality Scoring

The framework uses a multi-dimensional scoring system:

- **Contactability** (50%): Phone, email, website availability and validity
- **Completeness** (20%): Presence of key business data fields
- **Unicity** (20%): Duplicate detection across identifiers
- **Freshness** (10%): Data recency based on timestamps

Scores range from 0.0 to 1.0, with confidence levels:
- **High** (≥ 0.8): Ready for use
- **Medium** (0.6-0.8): Good quality, minor issues
- **Low** (< 0.6): Requires review

## Configuration

Quality thresholds and weights can be configured:

```python
config = {
    'thresholds': {
        'validation_rate': 0.8,      # 80% of results should be valid
        'phone_coverage': 0.6,       # 60% should have phone numbers
        'email_coverage': 0.4,       # 40% should have emails
        'minimum_confidence': 0.5,   # Minimum confidence to be valid
    },
    'field_weights': {
        'phone': 0.3,
        'email': 0.3,
        'website': 0.2,
        'address': 0.2
    }
}
```

## Dashboard Integration

The framework exports quality data for dashboard visualization:

```json
{
  "summary": {
    "total_records": 100,
    "valid_records": 75,
    "validation_rate": 75.0,
    "average_score": 68.0
  },
  "confidence_distribution": {"high": 25, "medium": 50, "low": 25},
  "field_coverage": {"phone": 80.0, "email": 60.0, "website": 40.0},
  "common_issues": [["Missing phone", 20], ["Invalid email", 15]],
  "recommendations": ["Improve phone coverage", "Validate email formats"]
}
```

## Testing

Comprehensive test suite with 47 tests covering:

- Contact extraction and validation
- Quality scoring algorithms
- Filtering and reporting
- Integration with Apify workflow

Run tests with:
```bash
python -m pytest tests/test_contact_extractor.py tests/test_quality_controller.py tests/test_apify_quality_integration.py -v
```

## Demonstration

See the complete demonstration of all features:
```bash
python demo_quality_framework.py
```

## Integration with Existing Pipeline

The quality framework integrates with the existing pipeline through:

1. **Input Preparation**: Enhanced address and company name processing
2. **Result Validation**: Real-time quality scoring during extraction
3. **Automated Filtering**: Quality-based result filtering
4. **Quality Reporting**: Dashboard data for monitoring and alerting

## File Structure

```
utils/
├── contact_extractor.py      # Contact extraction and validation
├── quality_controller.py     # Main quality control framework
api/
├── apify_agents.py           # Enhanced with quality validation
tests/
├── test_contact_extractor.py       # Contact extractor tests
├── test_quality_controller.py      # Quality controller tests
└── test_apify_quality_integration.py # Integration tests
demo_quality_framework.py     # Complete demonstration
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `requests`: HTTP validation
- `dnspython`: DNS validation
- `re`: Regular expressions
- `json`: JSON handling

## Performance Considerations

- Contact extraction is optimized for French business data
- Validation rules are configurable for different use cases
- Deep validation (DNS/HTTP checks) can be disabled for performance
- Quality scores are cached to avoid recomputation

## Future Enhancements

- Machine learning-based quality scoring
- Additional data source validators
- Real-time quality monitoring alerts
- Integration with external validation services