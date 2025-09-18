# Implementation Guide

## Overview

This guide provides comprehensive documentation on the implementation of features and enhancements in the data scraping and enrichment pipeline.

## Required Dataset Fields Implementation

### Problem Statement Requirements Fulfilled

The task was to add new information collection methods for standard and extended mode scrapping, ensuring all mandatory fields are present in the final `dataset.csv`.

### Required Fields (All Implemented)

| Field | Description | Source | Implementation |
|-------|-------------|--------|----------------|
| **Company name** | Business denomination | `denominationUniteLegale` | ✅ Mapped to `denomination` |
| **SIREN** | Business identifier | `siren` | ✅ Already existed |
| **Forme juridique** | Legal form | `categorieJuridiqueUniteLegale` | ✅ Added new column group |
| **Date d'immatriculation** | Registration date | `dateCreationUniteLegale` | ✅ Added new column group |
| **Adresse complète** | Complete address | `.parquet` files | ✅ Already existed |
| **Département** | From postal code (2 digits) | Extracted from postal code | ✅ New extraction logic |
| **Téléphone** | Phone via Google Maps | Google Maps scraping | ✅ Already existed |
| **Email** | Email via Google/Google Maps | Google Maps + heuristics | ✅ Already existed |
| **Site web** | Website via Google Maps | Google Maps scraping | ✅ Already existed |
| **Nom du dirigeant** | Director name | `nomUsageUniteLegale` + Google Maps | ✅ New extraction + Maps enhancement |

## Technical Implementation

### 1. Enhanced Data Normalization (`normalize/standardize.py`)

**Added new column groups:**
```python
'forme_juridique': [
    'categorieJuridiqueUniteLegale', 'formejuridique', 'FORMEJURIDIQUE'
],
'date_immatriculation': [
    'dateCreationUniteLegale', 'dateImmatriculation', 'date_creation'
],
'dirigeant_nom': [
    'nomUsageUniteLegale', 'dirigeant', 'DIRIGEANT'
],
'dirigeant_prenom': [
    'prenomUsuelUniteLegale', 'dirigeant_prenom'
]
```

**Added département extraction:**
- Handles both 5-digit (`06400`) and 4-digit (`6400`) postal codes
- Correctly extracts department codes with leading zeros (e.g., `06`, `01`)
- Robust logic for edge cases

### 2. Enhanced Google Maps Scraping (`enrich/google_maps_search.py`)

**Added director name extraction:**
```python
def extract_director_name(self, element):
    """Extract director/manager name from business listing"""
    # Look for common patterns like "Dirigeant: John Doe"
    text = element.get_text(strip=True)
    patterns = [
        r'(?:Dirigeant|Gérant|Président|Directeur|Manager):\s*([^,\n]+)',
        r'(?:dirigé par|géré par|sous la direction de)\s+([^,\n]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None
```

### 3. Google Maps Integration

The Google Maps enrichment module adds 9 new columns to the dataset:
- `gm_phone`: Phone number from Google Maps
- `gm_website`: Website URL from Google Maps  
- `gm_email`: Email address discovered through Google Maps
- `gm_hours`: Business hours
- `gm_rating`: Customer rating
- `gm_reviews`: Number of reviews
- `gm_address`: Confirmed address from Google Maps
- `gm_category`: Business category from Google Maps
- `gm_director`: Director/manager name

### 4. Budget Improvements

**Significantly Increased Limits**
| Setting | Before | After | Increase |
|---------|--------|-------|----------|
| HTTP Requests | 750 | 2000 | +267% |
| HTTP Bytes | 15MB | 50MB | +233% |
| Time Budget | 45 min | 90 min | +100% |
| RAM | 2GB | 4GB | +100% |

## Professional Services Implementation

### Generated Templates by Category

#### 🧮 Accounting & Financial Services
- `naf_6920Z.yaml` - Activités comptables (Expert-comptables)
- `naf_6622Z.yaml` - Agents et courtiers d'assurances
- `naf_6619A.yaml` - Gestion de patrimoine mobilier

#### ⚖️ Legal Services  
- `naf_6910Z.yaml` - Activités juridiques (Avocats, Notaires)
- `naf_6922Z.yaml` - Conseils pour les affaires

#### 💻 IT & Web Development
- `naf_6201Z.yaml` - Programmation informatique
- `naf_6202A.yaml` - Conseil en systèmes informatiques
- `naf_6202B.yaml` - Maintenance informatique

#### 📢 Marketing & Advertising
- `naf_7311Z.yaml` - Agences de publicité
- `naf_7312Z.yaml` - Régie publicitaire
- `naf_7320Z.yaml` - Études de marché et sondages
- `naf_7021Z.yaml` - Conseil en communication

#### 🏗️ Architecture & Engineering
- `naf_7111Z.yaml` - Activités d'architecture
- `naf_7112A.yaml` - Géomètres
- `naf_7112B.yaml` - Ingénierie et études techniques

#### 🏠 Real Estate
- `naf_6831Z.yaml` - Agences immobilières
- `naf_6832A.yaml` - Administration d'immeubles
- `naf_6832B.yaml` - Gestion de patrimoine immobilier

#### 💼 Business Consulting
- `naf_7022Z.yaml` - Conseil pour les affaires

#### 🎨 Creative Services
- `naf_7410Z.yaml` - Activités de design
- `naf_7420Z.yaml` - Activités photographiques

### Quality Standards

All professional services templates include:
- ✅ **Complete NAF Code Coverage**: 27 job templates covering all major categories
- ✅ **Accurate Industry Mapping**: Each NAF code mapped to relevant industry websites
- ✅ **Optimized for Web Scraping**: Specific seed URLs and domain configurations
- ✅ **Professional Quality Standards**: Enhanced budgets and quality targets

## Best Practices

### Data Collection
1. Use appropriate processing profiles (quick/standard/deep) based on needs
2. Set realistic budget limits for your infrastructure
3. Implement proper error handling for web scraping
4. Respect robots.txt and rate limiting

### Quality Control
1. Monitor KPI targets during processing
2. Validate extracted data with quality checks
3. Use duplicate detection and removal
4. Implement data freshness checks

### Performance Optimization
1. Use parallel processing with appropriate worker counts
2. Implement efficient caching strategies  
3. Monitor memory usage and implement limits
4. Use incremental processing for large datasets

## Troubleshooting

### Common Issues
- **Memory Issues**: Reduce workers, set RAM limits, use sampling
- **Network Issues**: Configure proxies, reduce request rates
- **Budget Exceeded**: Increase limits or optimize job configuration
- **Data Quality**: Review extraction patterns and validation rules

For detailed troubleshooting, see the main [README.md](../README.md) troubleshooting section.