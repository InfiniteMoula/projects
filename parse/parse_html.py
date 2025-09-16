import os
import re
import glob
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from utils import io

# Regular expressions for extracting contact information
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
SIRET_REGEX = re.compile(r'\b\d{14}\b')
SIREN_REGEX = re.compile(r'\b\d{9}\b')

def run(cfg, ctx):
    """Parse HTML files to extract structured contact and business information."""
    
    # Check for HTML files in both http and headless directories
    html_dirs = []
    for dirname in ["http", "headless"]:
        dir_path = os.path.join(ctx["outdir"], dirname)
        if os.path.isdir(dir_path):
            html_dirs.append(dir_path)
    
    if not html_dirs:
        return {"status": "SKIPPED", "reason": "NO_HTML_DIRS"}
    
    out = os.path.join(ctx["outdir"], "parsed_html")
    io.ensure_dir(out)
    
    extracted_data = []
    file_count = 0
    
    for html_dir in html_dirs:
        for filepath in glob.glob(os.path.join(html_dir, "*.html")):
            try:
                file_count += 1
                content = io.read_text(filepath)
                extracted = _extract_from_html(content, filepath)
                if extracted:
                    extracted_data.append(extracted)
                    
            except Exception as e:
                if ctx.get("logger"):
                    ctx["logger"].warning(f"Failed to parse {filepath}: {e}")
                continue
    
    # Save extracted data
    output_path = os.path.join(out, "extracted_contacts.json")
    io.write_json(output_path, {
        "data": extracted_data,
        "stats": {
            "files_processed": file_count,
            "records_extracted": len(extracted_data)
        }
    })
    
    return {
        "status": "OK",
        "file": output_path,
        "stats": {
            "files_processed": file_count,
            "records_extracted": len(extracted_data)
        }
    }

def _extract_from_html(content, filepath):
    """Extract structured data from HTML content."""
    
    soup = BeautifulSoup(content, 'html.parser')
    filename = os.path.basename(filepath)
    
    # Extract basic metadata
    result = {
        "source_file": filename,
        "title": _get_page_title(soup),
        "domain": _extract_domain_from_filename(filename),
        "emails": [],
        "phones": [],
        "addresses": [],
        "business_info": {},
        "social_links": [],
        "structured_data": []
    }
    
    # Extract emails
    text_content = soup.get_text()
    emails = EMAIL_REGEX.findall(text_content)
    result["emails"] = list(set(emails))  # Remove duplicates
    
    # Extract phone numbers
    phones = PHONE_REGEX.findall(text_content)
    result["phones"] = list(set(phones))
    
    # Extract business identifiers
    sirens = SIREN_REGEX.findall(text_content)
    sirets = SIRET_REGEX.findall(text_content)
    result["business_info"]["sirens"] = list(set(sirens))
    result["business_info"]["sirets"] = list(set(sirets))
    
    # Extract contact forms and mailto links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('mailto:'):
            email = href.replace('mailto:', '')
            if email not in result["emails"]:
                result["emails"].append(email)
        elif any(social in href.lower() for social in ['linkedin', 'twitter', 'facebook', 'instagram']):
            result["social_links"].append(href)
    
    # Extract structured data (JSON-LD, microdata, etc.)
    for script in soup.find_all('script', type='application/ld+json'):
        if script.string:
            try:
                data = json.loads(script.string)
                result["structured_data"].append(data)
            except json.JSONDecodeError:
                pass
    
    # Look for contact information in common patterns
    contact_keywords = ['contact', 'email', 'telephone', 'phone', 'adresse', 'address']
    for keyword in contact_keywords:
        elements = soup.find_all(text=re.compile(keyword, re.I))
        for element in elements:
            parent = element.parent if element.parent else None
            if parent:
                parent_text = parent.get_text(strip=True)
                # Extract additional contact info from context
                potential_emails = EMAIL_REGEX.findall(parent_text)
                potential_phones = PHONE_REGEX.findall(parent_text)
                result["emails"].extend(potential_emails)
                result["phones"].extend(potential_phones)
    
    # Remove duplicates
    result["emails"] = list(set(result["emails"]))
    result["phones"] = list(set(result["phones"]))
    
    # Only return if we found some useful data
    if (result["emails"] or result["phones"] or result["business_info"]["sirens"] 
        or result["business_info"]["sirets"] or result["structured_data"]):
        return result
    
    return None

def _get_page_title(soup):
    """Extract page title."""
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else ""

def _extract_domain_from_filename(filename):
    """Extract domain from filename."""
    # Assuming filenames are in format: domain.com_path.html
    parts = filename.split('_')
    if parts and '.' in parts[0]:
        return parts[0]
    return ""
