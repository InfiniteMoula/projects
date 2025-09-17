# FILE: parse/parse_jsonld.py
import os
import glob
import json
import re
import logging
from bs4 import BeautifulSoup
from utils import io

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')

def run(cfg, ctx):
    """Extract and parse JSON-LD structured data from HTML files."""
    
    # Check for HTML files in multiple directories
    html_dirs = []
    for dirname in ["http", "headless", "sitemaps"]:
        dir_path = os.path.join(ctx["outdir"], dirname)
        if os.path.isdir(dir_path):
            html_dirs.append(dir_path)
    
    if not html_dirs:
        return {"status": "SKIPPED", "reason": "NO_HTML"}
    
    out = os.path.join(ctx["outdir"], "jsonld")
    io.ensure_dir(out)
    
    extracted_data = []
    file_count = 0
    jsonld_count = 0
    
    for html_dir in html_dirs:
        for filepath in glob.glob(os.path.join(html_dir, "*.html")):
            try:
                file_count += 1
                content = io.read_text(filepath, encoding="utf-8")
                soup = BeautifulSoup(content, "lxml")
                
                filename = os.path.basename(filepath)
                
                for script_tag in soup.select('script[type="application/ld+json"]'):
                    # Use get_text() as fallback for script_tag.string
                    script_content = script_tag.string or script_tag.get_text()
                    if script_content:
                        # Strip HTML comments before JSON parsing
                        script_content = re.sub(r'<!--.*?-->', '', script_content, flags=re.DOTALL).strip()
                        
                        try:
                            data = json.loads(script_content)
                            jsonld_count += 1
                            
                            # Extract useful information from JSON-LD
                            extracted = _extract_from_jsonld(data, filename)
                            if extracted:
                                extracted_data.append(extracted)
                                
                        except json.JSONDecodeError:
                            # Fallback: extract emails and phones from the script text
                            emails = EMAIL_REGEX.findall(script_content)
                            phones = PHONE_REGEX.findall(script_content)
                            
                            if emails or phones:
                                fallback_record = {
                                    "source_file": filename,
                                    "type": "Fallback",
                                    "name": None,
                                    "emails": list(set(emails)) if emails else [],
                                    "phones": list(set(phones)) if phones else [],
                                    "addresses": [],
                                    "social_profiles": [],
                                    "website": None,
                                    "raw_data": {"fallback_text": script_content}
                                }
                                extracted_data.append(fallback_record)
                            
            except Exception as e:
                if ctx.get("logger"):
                    ctx["logger"].warning(f"Failed to parse {filepath}: {e}")
                continue
    
    # Save extracted data
    output_path = os.path.join(out, "extracted.json")
    result_data = {
        "data": extracted_data,
        "stats": {
            "files_processed": file_count,
            "jsonld_found": jsonld_count,
            "records_extracted": len(extracted_data)
        }
    }
    
    io.write_json(output_path, result_data)
    
    return {
        "file": output_path,
        "count": jsonld_count,
        "extracted_records": len(extracted_data),
        "status": "OK"
    }

def _extract_from_jsonld(data, source_file):
    """Extract contact and business information from JSON-LD data."""
    
    result = {
        "source_file": source_file,
        "type": None,
        "name": None,
        "emails": [],
        "phones": [],
        "addresses": [],
        "social_profiles": [],
        "website": None,
        "raw_data": data
    }
    
    # Handle arrays of data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _extract_from_object(item, result)
    elif isinstance(data, dict):
        _extract_from_object(data, result)
    
    # Clean up and deduplicate while preserving order for addresses
    result["emails"] = list(dict.fromkeys(result["emails"]))  # Preserve order while deduplicating
    result["phones"] = list(dict.fromkeys(result["phones"]))  # Preserve order while deduplicating
    result["social_profiles"] = list(dict.fromkeys(result["social_profiles"]))  # Preserve order while deduplicating
    
    # For addresses, preserve order and deduplicate
    seen_addresses = set()
    unique_addresses = []
    for addr in result["addresses"]:
        if addr not in seen_addresses:
            seen_addresses.add(addr)
            unique_addresses.append(addr)
    result["addresses"] = unique_addresses
    
    # Only return if we found useful contact data
    if (result["emails"] or result["phones"] or result["addresses"] 
        or result["website"] or result["name"]):
        return result
    
    return None

def _extract_from_object(obj, result):
    """Extract information from a JSON-LD object."""
    
    if not isinstance(obj, dict):
        return
    
    # Extract type - handle arrays
    obj_type = obj.get("@type", "")
    if obj_type and not result["type"]:
        result["type"] = obj_type
    
    # Extract name
    name_fields = ["name", "legalName", "alternateName"]
    for field in name_fields:
        if field in obj and not result["name"]:
            result["name"] = obj[field]
            break
    
    # Extract contact information
    contact_fields = ["email", "contactPoint", "contact"]
    for field in contact_fields:
        if field in obj:
            _extract_contact_info(obj[field], result)
    
    # Extract phone
    phone_fields = ["telephone", "phone", "phoneNumber"]
    for field in phone_fields:
        if field in obj:
            phone = obj[field]
            if isinstance(phone, str):
                result["phones"].append(phone)
            elif isinstance(phone, list):
                for p in phone:
                    if isinstance(p, str):
                        result["phones"].append(p)
    
    # Extract address - improved handling for nested objects
    address_fields = ["address", "location"]
    for field in address_fields:
        if field in obj:
            address_data = obj[field]
            if isinstance(address_data, list):
                for addr in address_data:
                    address_str = _format_address(addr)
                    if address_str:
                        result["addresses"].append(address_str)
            else:
                address_str = _format_address(address_data)
                if address_str:
                    result["addresses"].append(address_str)
    
    # Extract website/URL
    url_fields = ["url", "website", "sameAs"]
    for field in url_fields:
        if field in obj:
            url = obj[field]
            if isinstance(url, str) and url.startswith("http"):
                if not result["website"]:
                    result["website"] = url
                elif "linkedin" in url.lower() or "twitter" in url.lower() or "facebook" in url.lower():
                    result["social_profiles"].append(url)
            elif isinstance(url, list):
                for u in url:
                    if isinstance(u, str) and u.startswith("http"):
                        if not result["website"]:
                            result["website"] = u
                        elif any(social in u.lower() for social in ["linkedin", "twitter", "facebook", "instagram"]):
                            result["social_profiles"].append(u)
    
    # Recursively process nested objects
    for key, value in obj.items():
        if isinstance(value, dict):
            _extract_from_object(value, result)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _extract_from_object(item, result)

def _extract_contact_info(contact_data, result):
    """Extract contact information from various formats."""
    
    if isinstance(contact_data, str):
        # Direct email/phone string
        if "@" in contact_data:
            result["emails"].append(contact_data)
        elif PHONE_REGEX.match(contact_data):
            result["phones"].append(contact_data)
    elif isinstance(contact_data, dict):
        # Structured contact info
        if "email" in contact_data:
            result["emails"].append(contact_data["email"])
        if "telephone" in contact_data:
            result["phones"].append(contact_data["telephone"])
        if "phone" in contact_data:
            result["phones"].append(contact_data["phone"])
    elif isinstance(contact_data, list):
        # Array of contact info
        for item in contact_data:
            _extract_contact_info(item, result)

def _format_address(address_data):
    """Format address data into a string."""
    
    if isinstance(address_data, str):
        return address_data
    elif isinstance(address_data, dict):
        # Handle nested address objects (e.g., location.address)
        if "address" in address_data and isinstance(address_data["address"], dict):
            address_data = address_data["address"]
        
        parts = []
        address_fields = ["streetAddress", "addressLocality", "postalCode", "addressCountry"]
        for field in address_fields:
            if field in address_data and address_data[field]:
                parts.append(str(address_data[field]).strip())
        return ", ".join(parts) if parts else None
    elif isinstance(address_data, list):
        # Handle arrays of addresses
        addresses = []
        for addr in address_data:
            formatted = _format_address(addr)
            if formatted:
                addresses.append(formatted)
        return addresses[0] if addresses else None  # Return first valid address
    
    return None
