import os
import re
import glob
from utils import io

# Import PDF processing libraries
try:
    from pdfminer.high_level import extract_text
    from pdfminer.pdfpage import PDFPage
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Regular expressions for extracting contact information
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
SIRET_REGEX = re.compile(r'\b\d{14}\b')
SIREN_REGEX = re.compile(r'\b\d{9}\b')

def run(cfg, ctx):
    """Parse PDF files to extract contact and business information."""
    
    if not PDF_AVAILABLE:
        return {"status": "SKIPPED", "reason": "PDF_LIBRARY_NOT_AVAILABLE"}
    
    # Check for PDF files
    pdf_dirs = []
    for dirname in ["pdf", "pdfs"]:
        dir_path = os.path.join(ctx["outdir"], dirname)
        if os.path.isdir(dir_path):
            pdf_dirs.append(dir_path)
    
    if not pdf_dirs:
        return {"status": "SKIPPED", "reason": "NO_PDF_DIRS"}
    
    out = os.path.join(ctx["outdir"], "parsed_pdf")
    io.ensure_dir(out)
    
    extracted_data = []
    file_count = 0
    
    for pdf_dir in pdf_dirs:
        for filepath in glob.glob(os.path.join(pdf_dir, "*.pdf")):
            try:
                file_count += 1
                extracted = _extract_from_pdf(filepath)
                if extracted:
                    extracted_data.append(extracted)
                    
            except Exception as e:
                if ctx.get("logger"):
                    ctx["logger"].warning(f"Failed to parse {filepath}: {e}")
                continue
    
    # Save extracted data
    output_path = os.path.join(out, "extracted_pdf_data.json")
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

def _extract_from_pdf(filepath):
    """Extract structured data from PDF content."""
    
    filename = os.path.basename(filepath)
    
    try:
        # Extract text from PDF
        text_content = extract_text(filepath)
        
        if not text_content or len(text_content.strip()) < 10:
            return None
        
        # Initialize result structure
        result = {
            "source_file": filename,
            "text_length": len(text_content),
            "emails": [],
            "phones": [],
            "business_info": {},
            "keywords": []
        }
        
        # Extract emails
        emails = EMAIL_REGEX.findall(text_content)
        result["emails"] = list(set(emails))
        
        # Extract phone numbers
        phones = PHONE_REGEX.findall(text_content)
        result["phones"] = list(set(phones))
        
        # Extract business identifiers
        sirens = SIREN_REGEX.findall(text_content)
        sirets = SIRET_REGEX.findall(text_content)
        result["business_info"]["sirens"] = list(set(sirens))
        result["business_info"]["sirets"] = list(set(sirets))
        
        # Extract common business keywords
        business_keywords = [
            'société', 'sarl', 'sas', 'eurl', 'sa', 'sasu',
            'entreprise', 'cabinet', 'bureau', 'agence',
            'contact', 'directeur', 'gérant', 'président'
        ]
        
        text_lower = text_content.lower()
        found_keywords = [kw for kw in business_keywords if kw in text_lower]
        result["keywords"] = found_keywords
        
        # Extract potential addresses (simplified)
        address_patterns = [
            r'\d+\s+(?:rue|avenue|boulevard|place|chemin|impasse)[^,\n]*',
            r'\d{5}\s+[A-Za-z\-\s]+(?:France)?'
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            addresses.extend(matches)
        
        result["addresses"] = addresses[:5]  # Limit to first 5 addresses
        
        # Only return if we found useful data
        if (result["emails"] or result["phones"] or result["business_info"]["sirens"] 
            or result["business_info"]["sirets"] or result["addresses"]):
            return result
            
    except Exception as e:
        # Log error but don't fail
        return None
    
    return None
