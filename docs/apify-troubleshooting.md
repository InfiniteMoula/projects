# Apify Scrapers - Troubleshooting & Best Practices

This guide provides comprehensive troubleshooting solutions and best practices for optimizing Apify scrapers performance and reliability.

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Performance Optimization](#performance-optimization)
3. [Cost Management Best Practices](#cost-management-best-practices)
4. [Data Quality Best Practices](#data-quality-best-practices)
5. [Debugging and Monitoring](#debugging-and-monitoring)
6. [Production Deployment](#production-deployment)
7. [Maintenance and Updates](#maintenance-and-updates)

## Common Issues and Solutions

### 1. Authentication and API Issues

#### Issue: "APIFY_API_TOKEN environment variable is required"

**Cause**: Missing or incorrect API token configuration

**Solutions**:
```bash
# Check if token is set
echo $APIFY_API_TOKEN

# Set token in .env file
echo "APIFY_API_TOKEN=your_token_here" >> .env

# Set token for current session
export APIFY_API_TOKEN=your_token_here

# Verify token validity
python -c "
from apify_client import ApifyClient
import os
try:
    client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    user = client.user().get()
    print(f'✓ Token valid for user: {user.get(\"username\", \"unknown\")}')
except Exception as e:
    print(f'✗ Token invalid: {e}')
"
```

#### Issue: "Insufficient credits" or "Monthly usage exceeded"

**Cause**: Apify account has insufficient credits

**Solutions**:
```python
# Check current usage
def check_apify_usage():
    from apify_client import ApifyClient
    import os
    
    client = ApifyClient(os.getenv('APIFY_API_TOKEN'))
    user_info = client.user().get()
    
    current_usage = user_info.get('monthlyUsage', {}).get('credits', 0)
    monthly_limit = user_info.get('plan', {}).get('monthlyCredits', 0)
    
    print(f"Current usage: {current_usage} credits")
    print(f"Monthly limit: {monthly_limit} credits")
    print(f"Remaining: {monthly_limit - current_usage} credits")
    
    if current_usage >= monthly_limit * 0.9:
        print("⚠️ Warning: Close to credit limit")
    
    return {
        'current': current_usage,
        'limit': monthly_limit,
        'remaining': monthly_limit - current_usage
    }
```

**Prevention**:
- Monitor usage before large runs
- Set conservative limits in configuration
- Use cost estimation before processing

### 2. Data Input Issues

#### Issue: "No input data found" or "No addresses found"

**Cause**: Missing or incorrectly formatted input data

**Diagnostic Steps**:
```python
def diagnose_input_data(outdir):
    """Diagnose input data issues."""
    
    from pathlib import Path
    import pandas as pd
    
    outdir = Path(outdir)
    
    # Check for database.csv (step 7 output)
    database_path = outdir / "database.csv"
    if database_path.exists():
        try:
            df = pd.read_csv(database_path)
            print(f"✓ Found database.csv with {len(df)} rows")
            
            # Check for address column
            if 'adresse' in df.columns:
                valid_addresses = df['adresse'].dropna()
                print(f"✓ Found {len(valid_addresses)} valid addresses")
                
                # Show sample addresses
                print("Sample addresses:")
                for addr in valid_addresses.head(3):
                    print(f"  - {addr}")
            else:
                print("✗ No 'adresse' column found")
                print(f"Available columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"✗ Error reading database.csv: {e}")
    else:
        print("✗ database.csv not found")
    
    # Check for normalized.parquet (fallback)
    normalized_path = outdir / "normalized.parquet"
    if normalized_path.exists():
        try:
            df = pd.read_parquet(normalized_path)
            print(f"✓ Found normalized.parquet with {len(df)} rows")
        except Exception as e:
            print(f"✗ Error reading normalized.parquet: {e}")
    else:
        print("✗ normalized.parquet not found")
```

**Solutions**:
```bash
# Ensure step 7 (address extraction) runs successfully
python builder_cli.py run-step \
  --step enrich.address \
  --job your_job.yaml \
  --input data/input.parquet \
  --out out/test

# Check if address extraction worked
ls -la out/test/database.csv

# If using normalized data, ensure normalization step ran
python builder_cli.py run-step \
  --step normalize.standardize \
  --job your_job.yaml \
  --input data/input.parquet \
  --out out/test
```

#### Issue: Poor address quality leading to no results

**Cause**: Addresses are incomplete, poorly formatted, or invalid

**Solution - Address Quality Checker**:
```python
def check_address_quality(addresses):
    """Analyze address quality and suggest improvements."""
    
    import re
    
    quality_report = {
        'total_addresses': len(addresses),
        'quality_issues': {},
        'suggestions': []
    }
    
    for i, addr in enumerate(addresses):
        issues = []
        
        # Check length
        if len(addr) < 10:
            issues.append('too_short')
        elif len(addr) > 200:
            issues.append('too_long')
        
        # Check for postal code
        if not re.search(r'\b\d{5}\b', addr):
            issues.append('no_postal_code')
        
        # Check for street number
        if not re.search(r'^\d+', addr.strip()):
            issues.append('no_street_number')
        
        # Check for common street indicators
        street_indicators = ['rue', 'avenue', 'boulevard', 'place', 'impasse']
        if not any(indicator in addr.lower() for indicator in street_indicators):
            issues.append('no_street_type')
        
        if issues:
            quality_report['quality_issues'][i] = {
                'address': addr,
                'issues': issues
            }
    
    # Generate suggestions
    issue_counts = {}
    for addr_issues in quality_report['quality_issues'].values():
        for issue in addr_issues['issues']:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    if issue_counts.get('no_postal_code', 0) > len(addresses) * 0.3:
        quality_report['suggestions'].append("Many addresses missing postal codes - consider data cleaning")
    
    if issue_counts.get('too_short', 0) > len(addresses) * 0.2:
        quality_report['suggestions'].append("Many addresses too short - check address extraction process")
    
    return quality_report
```

### 3. Scraper-Specific Issues

#### Google Places Crawler Issues

**Issue**: "No places found" for valid addresses

**Debugging**:
```python
def debug_google_places_search(address, client):
    """Debug Google Places search issues."""
    
    print(f"Testing address: {address}")
    
    # Test different search variants
    variants = [
        address,
        address.replace(',', ''),  # Remove commas
        ' '.join(address.split()[1:]),  # Remove street number
        address.split(',')[-1].strip() if ',' in address else address  # Just city
    ]
    
    for i, variant in enumerate(variants):
        print(f"\nVariant {i+1}: {variant}")
        
        try:
            run_input = {
                "searchTerms": [variant],
                "language": "fr",
                "maxCrawledPlaces": 3
            }
            
            run = client.actor("compass/crawler-google-places").call(run_input=run_input)
            
            results = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            print(f"  Results: {len(results)} places found")
            
            for result in results[:2]:  # Show first 2 results
                print(f"    - {result.get('title', 'No title')}")
                
        except Exception as e:
            print(f"  Error: {e}")
```

**Common Solutions**:
- Use simplified address variants
- Try searches without street numbers
- Focus on city + postal code combinations
- Increase `maxCrawledPlaces` limit

#### LinkedIn Premium Actor Issues

**Issue**: "No profiles found" for valid companies

**Debugging**:
```python
def debug_linkedin_search(company_name, client):
    """Debug LinkedIn search issues."""
    
    print(f"Testing company: {company_name}")
    
    # Test different company name variants
    variants = [
        company_name,
        re.sub(r'\b(SAS|SARL|SA|EURL)\b', '', company_name, flags=re.IGNORECASE).strip(),
        company_name.split()[0] if len(company_name.split()) > 1 else company_name,
        company_name.upper(),
        company_name.title()
    ]
    
    for i, variant in enumerate(variants):
        if len(variant) < 3:
            continue
            
        print(f"\nVariant {i+1}: {variant}")
        
        try:
            run_input = {
                "searchTerms": [variant],
                "language": "fr",
                "maxProfiles": 3,
                "filters": {
                    "positions": ["CEO", "Directeur", "Gérant"]
                }
            }
            
            run = client.actor("bebity/linkedin-premium-actor").call(run_input=run_input)
            
            results = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            print(f"  Results: {len(results)} profiles found")
            
            for result in results[:2]:
                print(f"    - {result.get('fullName', 'No name')} - {result.get('position', 'No position')}")
                
        except Exception as e:
            print(f"  Error: {e}")
```

**Common Solutions**:
- Remove legal entity suffixes (SAS, SARL, etc.)
- Try abbreviated company names
- Use broader position filters
- Increase `maxProfiles` limit

### 4. Performance and Timeout Issues

#### Issue: "Actor timeout" errors

**Cause**: Scraper taking longer than expected

**Solutions**:
```yaml
# Increase timeouts in configuration
apify:
  google_places:
    timeout_seconds: 600  # 10 minutes (default: 300)
  
  google_maps_contacts:
    timeout_seconds: 450  # 7.5 minutes (default: 300)
  
  linkedin_premium:
    timeout_seconds: 900  # 15 minutes (default: 300)
```

**Monitoring Script**:
```python
def monitor_scraper_performance(client, actor_id, run_id):
    """Monitor scraper performance in real-time."""
    
    import time
    
    start_time = time.time()
    
    while True:
        try:
            run_info = client.actor(actor_id).run(run_id).get()
            status = run_info.get('status')
            
            elapsed = time.time() - start_time
            print(f"Status: {status}, Elapsed: {elapsed:.1f}s")
            
            if status in ['SUCCEEDED', 'FAILED', 'ABORTED']:
                break
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            break
    
    return run_info
```

## Performance Optimization

### 1. Batch Size Optimization

**Finding Optimal Batch Size**:
```python
def find_optimal_batch_size(addresses, client):
    """Find optimal batch size for your data."""
    
    batch_sizes = [5, 10, 20, 30]
    performance_results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        test_addresses = addresses[:batch_size]
        
        start_time = time.time()
        
        try:
            run_input = {
                "searchTerms": test_addresses,
                "language": "fr",
                "maxCrawledPlaces": 5
            }
            
            run = client.actor("compass/crawler-google-places").call(run_input=run_input)
            results = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            
            elapsed_time = time.time() - start_time
            success_rate = len(results) / batch_size if batch_size > 0 else 0
            
            performance_results[batch_size] = {
                'time': elapsed_time,
                'success_rate': success_rate,
                'time_per_address': elapsed_time / batch_size,
                'results_count': len(results)
            }
            
            print(f"  Time: {elapsed_time:.1f}s, Success: {success_rate:.1%}")
            
        except Exception as e:
            print(f"  Error: {e}")
            performance_results[batch_size] = {'error': str(e)}
    
    # Find optimal batch size
    valid_results = {k: v for k, v in performance_results.items() if 'error' not in v}
    
    if valid_results:
        optimal = min(valid_results.keys(), 
                     key=lambda x: valid_results[x]['time_per_address'])
        print(f"\nOptimal batch size: {optimal}")
        
    return performance_results
```

### 2. Memory Optimization

**Memory-Efficient Processing**:
```python
def process_large_dataset_efficiently(df, batch_size=50):
    """Process large datasets with memory optimization."""
    
    import gc
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    results = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing batch {batch_num + 1}/{total_batches}")
        
        # Process batch
        batch_results = process_apify_batch(batch_df)
        results.extend(batch_results)
        
        # Memory cleanup
        del batch_df, batch_results
        gc.collect()
        
        # Rate limiting
        if batch_num < total_batches - 1:
            time.sleep(2)
    
    return results
```

### 3. Parallel Processing Strategy

**Future Enhancement - Async Processing**:
```python
import asyncio
from typing import List, Dict

async def process_scrapers_parallel(addresses, companies, client, config):
    """Process multiple scrapers in parallel for better performance."""
    
    tasks = []
    
    # Google Places (foundation)
    if config.get("google_places", {}).get("enabled", True):
        tasks.append(run_google_places_async(addresses, client, config))
    
    # LinkedIn (independent of Google results)
    if config.get("linkedin_premium", {}).get("enabled", True) and companies:
        tasks.append(run_linkedin_premium_async(companies, client, config))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    places_results = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
    linkedin_results = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
    
    # Google Maps Contacts (depends on places results)
    contact_results = []
    if config.get("google_maps_contacts", {}).get("enabled", True) and places_results:
        contact_results = await run_google_maps_contacts_async(places_results, client, config)
    
    return {
        'places': places_results,
        'contacts': contact_results,
        'linkedin': linkedin_results
    }

async def run_google_places_async(addresses, client, config):
    """Async version of Google Places scraper."""
    # Implementation would use aiohttp or async Apify client
    pass
```

## Cost Management Best Practices

### 1. Pre-Processing Cost Estimation

**Cost Calculator**:
```python
class ApifyCostCalculator:
    """Calculate estimated costs before running scrapers."""
    
    def __init__(self):
        # Average credit costs per operation
        self.cost_estimates = {
            'google_places': {
                'cost_per_search': 3,  # 1-5 credits average
                'cost_per_result': 0.5
            },
            'google_maps_contacts': {
                'cost_per_enrichment': 2,  # 1-3 credits average
                'cost_per_result': 1
            },
            'linkedin_premium': {
                'cost_per_search': 30,  # 10-50 credits average
                'cost_per_profile': 6
            }
        }
    
    def estimate_total_cost(self, config: Dict, input_size: int) -> Dict:
        """Estimate total cost for configuration and input size."""
        
        costs = {}
        total_cost = 0
        
        max_addresses = min(config.get("max_addresses", input_size), input_size)
        
        # Google Places
        if config.get("google_places", {}).get("enabled", True):
            places_cost = max_addresses * self.cost_estimates['google_places']['cost_per_search']
            costs['google_places'] = places_cost
            total_cost += places_cost
        
        # Google Maps Contacts
        if config.get("google_maps_contacts", {}).get("enabled", True):
            max_enrichments = config.get("google_maps_contacts", {}).get("max_contact_enrichments", 50)
            estimated_enrichments = min(max_enrichments, max_addresses * 3)  # 3 places per address avg
            contacts_cost = estimated_enrichments * self.cost_estimates['google_maps_contacts']['cost_per_enrichment']
            costs['google_maps_contacts'] = contacts_cost
            total_cost += contacts_cost
        
        # LinkedIn Premium
        if config.get("linkedin_premium", {}).get("enabled", True):
            max_searches = config.get("linkedin_premium", {}).get("max_linkedin_searches", 20)
            linkedin_searches = min(max_searches, max_addresses)
            linkedin_cost = linkedin_searches * self.cost_estimates['linkedin_premium']['cost_per_search']
            costs['linkedin_premium'] = linkedin_cost
            total_cost += linkedin_cost
        
        return {
            'breakdown': costs,
            'total_estimated': total_cost,
            'addresses_processed': max_addresses,
            'cost_per_address': total_cost / max_addresses if max_addresses > 0 else 0,
            'recommendation': self._get_cost_recommendation(total_cost)
        }
    
    def _get_cost_recommendation(self, total_cost: int) -> str:
        """Get cost recommendation based on estimated total."""
        
        if total_cost < 50:
            return "LOW_COST - Safe for testing and small batches"
        elif total_cost < 200:
            return "MEDIUM_COST - Suitable for regular processing"
        elif total_cost < 500:
            return "HIGH_COST - Use for important business data"
        else:
            return "VERY_HIGH_COST - Consider reducing scope or batch processing"
```

### 2. Dynamic Budget Allocation

**Budget-Based Configuration**:
```python
def create_budget_optimized_config(available_credits: int, priorities: List[str]) -> Dict:
    """Create configuration optimized for available budget."""
    
    configs = {
        'minimal': {  # < 100 credits
            'max_addresses': 10,
            'google_places': {'enabled': True, 'max_places_per_search': 3},
            'google_maps_contacts': {'enabled': False},
            'linkedin_premium': {'enabled': False}
        },
        'basic': {  # 100-300 credits
            'max_addresses': 20,
            'google_places': {'enabled': True, 'max_places_per_search': 5},
            'google_maps_contacts': {'enabled': True, 'max_contact_enrichments': 15},
            'linkedin_premium': {'enabled': False}
        },
        'standard': {  # 300-800 credits
            'max_addresses': 30,
            'google_places': {'enabled': True, 'max_places_per_search': 8},
            'google_maps_contacts': {'enabled': True, 'max_contact_enrichments': 25},
            'linkedin_premium': {'enabled': True, 'max_linkedin_searches': 10, 'max_profiles_per_company': 3}
        },
        'premium': {  # 800+ credits
            'max_addresses': 50,
            'google_places': {'enabled': True, 'max_places_per_search': 10},
            'google_maps_contacts': {'enabled': True, 'max_contact_enrichments': 50},
            'linkedin_premium': {'enabled': True, 'max_linkedin_searches': 25, 'max_profiles_per_company': 5}
        }
    }
    
    # Select appropriate config based on budget
    if available_credits < 100:
        base_config = configs['minimal']
    elif available_credits < 300:
        base_config = configs['basic']
    elif available_credits < 800:
        base_config = configs['standard']
    else:
        base_config = configs['premium']
    
    # Adjust based on priorities
    if 'contacts' in priorities and available_credits >= 200:
        base_config['google_maps_contacts']['enabled'] = True
        if available_credits >= 400:
            base_config['google_maps_contacts']['max_contact_enrichments'] *= 2
    
    if 'executives' in priorities and available_credits >= 500:
        base_config['linkedin_premium']['enabled'] = True
        if available_credits >= 1000:
            base_config['linkedin_premium']['max_profiles_per_company'] += 2
    
    return base_config
```

## Data Quality Best Practices

### 1. Input Data Preparation

**Address Standardization**:
```python
class AddressStandardizer:
    """Standardize addresses for better scraper results."""
    
    def __init__(self):
        self.french_abbreviations = {
            'RUE': 'Rue',
            'AVE': 'Avenue',
            'BLVD': 'Boulevard',
            'BD': 'Boulevard',
            'PL': 'Place',
            'IMP': 'Impasse',
            'ALL': 'Allée',
            'QU': 'Quai',
            'SQ': 'Square'
        }
        
        self.postal_code_regex = re.compile(r'\b(\d{5})\b')
    
    def standardize_address(self, address: str) -> str:
        """Standardize single address."""
        
        if not address or pd.isna(address):
            return ""
        
        # Clean whitespace
        addr = re.sub(r'\s+', ' ', str(address).strip())
        
        # Standardize abbreviations
        for abbrev, full in self.french_abbreviations.items():
            addr = re.sub(rf'\b{abbrev}\b', full, addr, flags=re.IGNORECASE)
        
        # Ensure proper capitalization
        addr = self._proper_case_address(addr)
        
        # Validate postal code format
        addr = self._validate_postal_code(addr)
        
        return addr
    
    def _proper_case_address(self, address: str) -> str:
        """Apply proper case to address components."""
        
        words = address.split()
        result = []
        
        for word in words:
            # Keep postal codes as-is
            if re.match(r'^\d{5}$', word):
                result.append(word)
            # Keep small connecting words lowercase
            elif word.lower() in ['de', 'du', 'des', 'le', 'la', 'les']:
                result.append(word.lower())
            # Capitalize other words
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _validate_postal_code(self, address: str) -> str:
        """Validate and correct postal code format."""
        
        postal_matches = self.postal_code_regex.findall(address)
        
        for postal in postal_matches:
            # Check if it's a valid French postal code
            if postal.startswith('0') or postal.startswith('9'):
                # These are often errors or special cases
                continue
            
            # Ensure postal code is properly positioned
            # (This is a simplified validation)
        
        return address
```

**Company Name Optimization**:
```python
class CompanyNameOptimizer:
    """Optimize company names for LinkedIn searches."""
    
    def __init__(self):
        self.legal_forms = [
            'SAS', 'SARL', 'SA', 'EURL', 'SNC', 'SCS', 'SASU',
            'SELARL', 'SELAFA', 'SELAS', 'SELCA'
        ]
        
        self.common_suffixes = [
            '& Associés', '& Associates', '& Co', '& Cie',
            'Conseil', 'Consulting', 'Services', 'Solutions'
        ]
    
    def optimize_for_linkedin(self, company_name: str) -> Dict:
        """Create optimized variants for LinkedIn search."""
        
        if not company_name or pd.isna(company_name):
            return {'variants': [], 'confidence': 0.0}
        
        variants = []
        base_name = str(company_name).strip()
        
        # Original name
        variants.append(base_name)
        
        # Remove legal forms
        clean_name = self._remove_legal_forms(base_name)
        if clean_name != base_name:
            variants.append(clean_name)
        
        # Remove common suffixes
        no_suffix = self._remove_suffixes(clean_name)
        if no_suffix != clean_name:
            variants.append(no_suffix)
        
        # Create acronym for long names
        if len(clean_name.split()) > 3:
            acronym = self._create_acronym(clean_name)
            if acronym:
                variants.append(acronym)
        
        # Remove duplicates while preserving order
        unique_variants = []
        for variant in variants:
            if variant not in unique_variants and len(variant.strip()) > 2:
                unique_variants.append(variant)
        
        confidence = self._calculate_linkedin_confidence(base_name)
        
        return {
            'original': base_name,
            'variants': unique_variants,
            'primary': unique_variants[0] if unique_variants else base_name,
            'confidence': confidence
        }
    
    def _remove_legal_forms(self, name: str) -> str:
        """Remove legal entity forms from company name."""
        
        for form in self.legal_forms:
            # Remove as separate word
            pattern = rf'\b{re.escape(form)}\b'
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _calculate_linkedin_confidence(self, name: str) -> float:
        """Calculate confidence for LinkedIn search success."""
        
        confidence = 0.5  # Base confidence
        
        # Length scoring
        if 5 <= len(name) <= 50:
            confidence += 0.2
        
        # Has meaningful content after cleaning
        clean_name = self._remove_legal_forms(name)
        if len(clean_name) > 3:
            confidence += 0.2
        
        # Contains business-like words
        business_indicators = ['consulting', 'conseil', 'services', 'solutions', 'group', 'groupe']
        if any(indicator in name.lower() for indicator in business_indicators):
            confidence += 0.1
        
        return min(confidence, 1.0)
```

### 2. Result Validation and Scoring

**Comprehensive Result Validator**:
```python
class ResultValidator:
    """Validate and score extracted results."""
    
    def __init__(self):
        self.phone_pattern = re.compile(r'^(\+33|0)[1-9]\d{8}$')
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.website_pattern = re.compile(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
    def validate_result(self, result: Dict, source: str) -> Dict:
        """Validate a single extraction result."""
        
        validation = {
            'source': source,
            'field_scores': {},
            'overall_score': 0.0,
            'issues': [],
            'confidence': 'low'
        }
        
        # Validate each field based on source
        if source == 'google_places':
            validation.update(self._validate_google_places_result(result))
        elif source == 'google_maps_contacts':
            validation.update(self._validate_google_maps_result(result))
        elif source == 'linkedin_premium':
            validation.update(self._validate_linkedin_result(result))
        
        # Calculate overall score
        if validation['field_scores']:
            validation['overall_score'] = sum(validation['field_scores'].values()) / len(validation['field_scores'])
        
        # Determine confidence level
        if validation['overall_score'] >= 0.8:
            validation['confidence'] = 'high'
        elif validation['overall_score'] >= 0.6:
            validation['confidence'] = 'medium'
        else:
            validation['confidence'] = 'low'
        
        return validation
    
    def _validate_google_places_result(self, result: Dict) -> Dict:
        """Validate Google Places specific result."""
        
        field_scores = {}
        issues = []
        
        # Title validation
        title = result.get('title', '')
        if title:
            if len(title) >= 3 and not title.isupper():
                field_scores['title'] = 0.9
            else:
                field_scores['title'] = 0.5
                issues.append("Title format questionable")
        else:
            field_scores['title'] = 0.0
            issues.append("Missing title")
        
        # Phone validation
        phone = result.get('phone', '')
        if phone:
            if self.phone_pattern.match(re.sub(r'[\s.-]', '', phone)):
                field_scores['phone'] = 1.0
            else:
                field_scores['phone'] = 0.3
                issues.append("Invalid phone format")
        else:
            field_scores['phone'] = 0.0
        
        # Address validation
        address = result.get('address', '')
        if address and len(address) > 10:
            field_scores['address'] = 0.8
        else:
            field_scores['address'] = 0.2
            issues.append("Address too short or missing")
        
        return {'field_scores': field_scores, 'issues': issues}
    
    def _validate_linkedin_result(self, result: Dict) -> Dict:
        """Validate LinkedIn specific result."""
        
        field_scores = {}
        issues = []
        
        # Name validation
        full_name = result.get('fullName', '')
        if full_name:
            name_parts = full_name.strip().split()
            if len(name_parts) >= 2 and all(len(part) >= 2 for part in name_parts):
                field_scores['name'] = 0.9
            else:
                field_scores['name'] = 0.4
                issues.append("Name format questionable")
        else:
            field_scores['name'] = 0.0
            issues.append("Missing name")
        
        # Position validation
        position = result.get('position', '')
        executive_keywords = ['directeur', 'président', 'gérant', 'ceo', 'cfo', 'manager', 'fondateur']
        if position and any(keyword in position.lower() for keyword in executive_keywords):
            field_scores['position'] = 0.9
        elif position:
            field_scores['position'] = 0.5
            issues.append("Position not clearly executive level")
        else:
            field_scores['position'] = 0.0
            issues.append("Missing position")
        
        # Company match validation
        company_name = result.get('companyName', '')
        search_term = result.get('searchTerm', '')
        if company_name and search_term:
            from rapidfuzz import fuzz
            similarity = fuzz.partial_ratio(company_name.lower(), search_term.lower())
            if similarity > 80:
                field_scores['company_match'] = 0.9
            elif similarity > 60:
                field_scores['company_match'] = 0.6
                issues.append("Company match uncertain")
            else:
                field_scores['company_match'] = 0.3
                issues.append("Poor company match")
        else:
            field_scores['company_match'] = 0.0
            issues.append("Cannot verify company match")
        
        return {'field_scores': field_scores, 'issues': issues}
```

## Debugging and Monitoring

### 1. Comprehensive Logging

**Debug Logger Setup**:
```python
import logging
import sys
from pathlib import Path

def setup_apify_logging(debug_mode=False, log_file=None):
    """Setup comprehensive logging for Apify operations."""
    
    # Create logger
    logger = logging.getLogger('apify_scrapers')
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# Usage in apify_agents.py
logger = setup_apify_logging(debug_mode=True, log_file='apify_debug.log')

def run(cfg: dict, ctx: dict) -> dict:
    logger.info("Starting Apify processing")
    logger.debug(f"Configuration: {cfg}")
    
    # ... rest of implementation with comprehensive logging
```

### 2. Performance Monitoring

**Performance Tracker**:
```python
class ApifyPerformanceTracker:
    """Track performance metrics during scraping."""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'scraper_times': {},
            'error_counts': {},
            'success_rates': {},
            'cost_tracking': {}
        }
    
    def start_tracking(self):
        """Start performance tracking."""
        self.metrics['start_time'] = time.time()
    
    def track_scraper(self, scraper_name: str, start_time: float, success_count: int, total_count: int, estimated_cost: int):
        """Track individual scraper performance."""
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.metrics['scraper_times'][scraper_name] = duration
        self.metrics['success_rates'][scraper_name] = success_count / total_count if total_count > 0 else 0
        self.metrics['cost_tracking'][scraper_name] = estimated_cost
    
    def track_error(self, scraper_name: str, error: str):
        """Track errors by scraper."""
        
        if scraper_name not in self.metrics['error_counts']:
            self.metrics['error_counts'][scraper_name] = {}
        
        error_type = type(error).__name__ if isinstance(error, Exception) else 'unknown'
        self.metrics['error_counts'][scraper_name][error_type] = \
            self.metrics['error_counts'][scraper_name].get(error_type, 0) + 1
    
    def finish_tracking(self):
        """Finish tracking and generate report."""
        
        self.metrics['end_time'] = time.time()
        total_time = self.metrics['end_time'] - self.metrics['start_time']
        
        report = {
            'total_duration': total_time,
            'scraper_performance': {},
            'overall_stats': {}
        }
        
        # Scraper-specific performance
        for scraper, duration in self.metrics['scraper_times'].items():
            success_rate = self.metrics['success_rates'].get(scraper, 0)
            cost = self.metrics['cost_tracking'].get(scraper, 0)
            
            report['scraper_performance'][scraper] = {
                'duration': duration,
                'success_rate': success_rate,
                'estimated_cost': cost,
                'efficiency_score': success_rate / (duration / 60) if duration > 0 else 0  # success per minute
            }
        
        # Overall statistics
        total_cost = sum(self.metrics['cost_tracking'].values())
        avg_success_rate = sum(self.metrics['success_rates'].values()) / len(self.metrics['success_rates']) if self.metrics['success_rates'] else 0
        
        report['overall_stats'] = {
            'total_estimated_cost': total_cost,
            'average_success_rate': avg_success_rate,
            'total_errors': sum(sum(errors.values()) for errors in self.metrics['error_counts'].values()),
            'cost_per_minute': total_cost / (total_time / 60) if total_time > 0 else 0
        }
        
        return report
```

### 3. Real-Time Monitoring Dashboard

**Simple Dashboard Creation**:
```python
def create_monitoring_dashboard(performance_data):
    """Create simple text-based monitoring dashboard."""
    
    dashboard = []
    dashboard.append("=" * 60)
    dashboard.append("APIFY SCRAPERS PERFORMANCE DASHBOARD")
    dashboard.append("=" * 60)
    
    # Overall stats
    overall = performance_data.get('overall_stats', {})
    dashboard.append(f"Total Duration: {overall.get('total_duration', 0):.1f} seconds")
    dashboard.append(f"Average Success Rate: {overall.get('average_success_rate', 0):.1%}")
    dashboard.append(f"Total Estimated Cost: {overall.get('total_estimated_cost', 0)} credits")
    dashboard.append(f"Total Errors: {overall.get('total_errors', 0)}")
    dashboard.append("")
    
    # Scraper-specific performance
    dashboard.append("SCRAPER PERFORMANCE:")
    dashboard.append("-" * 40)
    
    for scraper, stats in performance_data.get('scraper_performance', {}).items():
        dashboard.append(f"{scraper.upper()}:")
        dashboard.append(f"  Duration: {stats.get('duration', 0):.1f}s")
        dashboard.append(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
        dashboard.append(f"  Cost: {stats.get('estimated_cost', 0)} credits")
        dashboard.append(f"  Efficiency: {stats.get('efficiency_score', 0):.2f} success/min")
        dashboard.append("")
    
    return "\n".join(dashboard)
```

## Production Deployment

### 1. Production Configuration

**Production-Ready Config Template**:
```yaml
# production_apify_config.yaml
niche: "production_scraping"

# Robust filtering for production
filters:
  naf_include: ["6920Z", "7022Z"]
  active_only: true
  regions: ["75", "92", "93", "94"]

profile: "standard"

# Production Apify settings
apify:
  enabled: true
  max_addresses: 100
  save_raw_results: false  # Save space in production
  
  # Conservative timeouts for reliability
  google_places:
    enabled: true
    max_places_per_search: 8
    timeout_seconds: 600
    retry_failed: true
  
  google_maps_contacts:
    enabled: true
    max_contact_enrichments: 80
    timeout_seconds: 450
    retry_failed: true
  
  linkedin_premium:
    enabled: true
    max_linkedin_searches: 50
    max_profiles_per_company: 4
    timeout_seconds: 900
    retry_failed: true

# Production budget controls
budget:
  time_budget_min: 180  # 3 hours max
  credit_limit: 2000    # Monthly credit limit
  alert_threshold: 0.8  # Alert at 80% usage

# Production monitoring
monitoring:
  enabled: true
  log_level: "INFO"
  error_threshold: 0.1  # Alert if error rate > 10%
  success_threshold: 0.7  # Alert if success rate < 70%
```

### 2. Error Recovery and Resilience

**Production Error Handler**:
```python
class ProductionErrorHandler:
    """Handle errors gracefully in production environment."""
    
    def __init__(self, config):
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 60)  # seconds
        self.error_log = []
    
    def handle_scraper_error(self, scraper_name: str, error: Exception, context: Dict) -> Dict:
        """Handle scraper errors with appropriate recovery strategy."""
        
        error_info = {
            'timestamp': datetime.now(),
            'scraper': scraper_name,
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context
        }
        
        self.error_log.append(error_info)
        
        # Determine recovery strategy based on error type
        if isinstance(error, (ConnectionError, TimeoutError)):
            return self._handle_connectivity_error(scraper_name, error, context)
        elif 'credit' in str(error).lower() or 'usage' in str(error).lower():
            return self._handle_credit_error(scraper_name, error, context)
        elif 'rate limit' in str(error).lower():
            return self._handle_rate_limit_error(scraper_name, error, context)
        else:
            return self._handle_generic_error(scraper_name, error, context)
    
    def _handle_connectivity_error(self, scraper_name: str, error: Exception, context: Dict) -> Dict:
        """Handle connectivity/timeout errors."""
        
        if context.get('retry_count', 0) < self.max_retries:
            return {
                'action': 'retry',
                'delay': self.retry_delay,
                'retry_count': context.get('retry_count', 0) + 1,
                'modified_config': self._reduce_batch_size(context.get('config', {}))
            }
        else:
            return {
                'action': 'skip',
                'reason': 'Max retries exceeded for connectivity issues'
            }
    
    def _handle_credit_error(self, scraper_name: str, error: Exception, context: Dict) -> Dict:
        """Handle credit/billing errors."""
        
        return {
            'action': 'disable_scraper',
            'reason': 'Insufficient credits',
            'recommendation': 'Check Apify account balance and increase credits'
        }
    
    def _reduce_batch_size(self, config: Dict) -> Dict:
        """Reduce batch size for retry attempt."""
        
        modified = config.copy()
        
        # Reduce limits by 50%
        if 'max_addresses' in modified:
            modified['max_addresses'] = max(1, modified['max_addresses'] // 2)
        
        for scraper in ['google_places', 'google_maps_contacts', 'linkedin_premium']:
            if scraper in modified:
                if 'max_places_per_search' in modified[scraper]:
                    modified[scraper]['max_places_per_search'] = max(1, modified[scraper]['max_places_per_search'] // 2)
                if 'max_contact_enrichments' in modified[scraper]:
                    modified[scraper]['max_contact_enrichments'] = max(1, modified[scraper]['max_contact_enrichments'] // 2)
        
        return modified
```

### 3. Health Checks and Monitoring

**Health Check System**:
```python
class ApifyHealthChecker:
    """Monitor Apify system health."""
    
    def __init__(self, client):
        self.client = client
        self.health_status = {}
    
    def check_api_health(self) -> Dict:
        """Check Apify API connectivity and account status."""
        
        try:
            user_info = self.client.user().get()
            
            # Check account status
            account_status = {
                'api_accessible': True,
                'username': user_info.get('username'),
                'plan': user_info.get('plan', {}).get('name'),
                'credits_remaining': self._get_remaining_credits(user_info),
                'account_active': user_info.get('isBlocked', True) == False
            }
            
            # Check actor availability
            actor_status = self._check_actor_availability()
            
            return {
                'overall_health': 'healthy' if account_status['api_accessible'] and account_status['account_active'] else 'unhealthy',
                'account': account_status,
                'actors': actor_status,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'overall_health': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _check_actor_availability(self) -> Dict:
        """Check if required actors are available."""
        
        required_actors = [
            'compass/crawler-google-places',
            'lukaskrivka/google-maps-with-contact-details',
            'bebity/linkedin-premium-actor'
        ]
        
        actor_status = {}
        
        for actor_id in required_actors:
            try:
                actor_info = self.client.actor(actor_id).get()
                actor_status[actor_id] = {
                    'available': True,
                    'name': actor_info.get('name'),
                    'last_build': actor_info.get('lastBuildDate'),
                    'is_public': actor_info.get('isPublic', False)
                }
            except Exception as e:
                actor_status[actor_id] = {
                    'available': False,
                    'error': str(e)
                }
        
        return actor_status
    
    def _get_remaining_credits(self, user_info: Dict) -> int:
        """Calculate remaining credits for the month."""
        
        monthly_usage = user_info.get('monthlyUsage', {}).get('credits', 0)
        monthly_limit = user_info.get('plan', {}).get('monthlyCredits', 0)
        
        return max(0, monthly_limit - monthly_usage)
```

This comprehensive troubleshooting and best practices guide provides the foundation for reliable, cost-effective, and high-quality Apify scraper operations in both development and production environments.