# Legal Compliance Guidelines

## Scraping Ethics and Compliance

### ✅ Required Compliance
- **Strict respect for robots.txt** directives on all websites
- **Terms of Service (ToS)** compliance for all data sources
- **X-Robots-Tag and meta robots** header respect
- **Rate limiting** to avoid overwhelming target servers
- **Proper user-agent identification** in all requests

### ❌ Prohibited Activities
- **SERP scraping** (Search Engine Results Pages)
- **Login-protected content** access without authorization
- **Paywall bypass** or circumvention attempts
- **CAPTCHA bypass** or automated solving
- **Simulated user actions** beyond legitimate data collection

## Data Sources and Licensing

### SIRENE Database
- **Source**: French government business registry
- **License**: Licence Ouverte (Open License)
- **Usage**: Permitted for commercial and non-commercial use
- **Attribution**: Required for derivative works

### Email Address Handling
- **Generated emails**: Plausible generic patterns only
- **No active verification**: No actual email sending or validation
- **Privacy compliance**: GDPR-compliant processing
- **Purpose limitation**: Business contact discovery only

## Compliance Reporting

### Manifest Documentation
The system generates `manifest.json` files with compliance indicators:
- `robots_compliance: true` - Confirms robots.txt respect
- `tos_breaches: []` - Lists any detected ToS violations
- `exceptions: []` - Documents any compliance exceptions encountered

### Audit Trail
All scraping activities are logged with:
- Target URLs and domains
- Robots.txt compliance status
- Rate limiting adherence
- Error conditions and responses

## Best Practices

1. **Always check robots.txt** before scraping any domain
2. **Implement conservative rate limits** (≤ 1 request per second)
3. **Use descriptive user agents** identifying your organization
4. **Monitor for blocking signals** (HTTP 429, 403 responses)
5. **Respect data subject rights** under GDPR and similar regulations

## Contact and Disputes

For questions about data usage or compliance concerns, maintain clear contact information and respond promptly to legitimate requests for data modification or removal.