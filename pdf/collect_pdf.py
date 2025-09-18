import os
import time
import hashlib
from urllib.parse import urlparse, urljoin
from utils import io
import httpx

def run(cfg, ctx):
    """
    Collect PDF documents from configured URLs.
    
    Configuration:
    pdf:
      urls: ["http://example.com/doc.pdf", "http://example.com/brochure.pdf"]
      max_size_mb: 10
      timeout: 60
    """
    
    pdf_config = cfg.get("pdf", {})
    pdf_urls = pdf_config.get("urls", [])
    max_size_mb = pdf_config.get("max_size_mb", 10)
    timeout = pdf_config.get("timeout", 60)
    
    if not pdf_urls:
        return {"status": "SKIPPED", "reason": "NO_PDF_URLS"}
    
    out = os.path.join(ctx["outdir"], "pdf")
    io.ensure_dir(out)
    
    collected_pdfs = []
    total_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024
    
    logger = ctx.get("logger")
    
    for pdf_url in pdf_urls[:20]:  # Limit to 20 PDFs for safety
        try:
            if logger:
                logger.info(f"Downloading PDF: {pdf_url}")
            
            result = _download_pdf(pdf_url, out, max_size_bytes, timeout, logger)
            if result:
                collected_pdfs.append(result)
                total_size += result["size_bytes"]
                
                # Safety check for total size
                if total_size > max_size_bytes * 5:  # 5x limit total
                    if logger:
                        logger.warning("Total PDF size limit exceeded, stopping collection")
                    break
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to download PDF {pdf_url}: {e}")
            continue
    
    return {
        "status": "OK",
        "pdfs_collected": len(collected_pdfs),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": collected_pdfs,
        "output_dir": out
    }

def _download_pdf(url, output_dir, max_size_bytes, timeout, logger=None):
    """Download a PDF file."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PDFBot/1.0)"
        }
        
        # Parse URL to create filename
        parsed = urlparse(url)
        domain = parsed.netloc or "unknown"
        path_parts = parsed.path.strip("/").split("/")
        
        # Create safe filename
        if path_parts and path_parts[-1].endswith('.pdf'):
            filename = path_parts[-1]
        else:
            filename = f"document_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
        
        # Ensure unique filename
        safe_filename = _make_safe_filename(f"{domain}_{filename}")
        file_path = os.path.join(output_dir, safe_filename)
        
        # Check if already exists
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not url.lower().endswith('.pdf'):
                    if logger:
                        logger.warning(f"URL may not be PDF: {url} (content-type: {content_type})")
                
                # Check content length
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size_bytes:
                    if logger:
                        logger.warning(f"PDF too large: {url} ({content_length} bytes)")
                    return None
                
                # Download with size checking
                total_size = 0
                with open(file_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        total_size += len(chunk)
                        if total_size > max_size_bytes:
                            f.close()
                            os.remove(file_path)
                            if logger:
                                logger.warning(f"PDF download too large: {url}")
                            return None
                        f.write(chunk)
                
                # Verify it's actually a PDF by checking magic bytes
                if not _is_pdf_file(file_path):
                    os.remove(file_path)
                    if logger:
                        logger.warning(f"Downloaded file is not a valid PDF: {url}")
                    return None
                
                return {
                    "url": url,
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "size_bytes": total_size,
                    "domain": domain
                }
                
    except Exception as e:
        if logger:
            logger.error(f"Error downloading PDF {url}: {e}")
        return None

def _make_safe_filename(filename):
    """Make filename safe for filesystem."""
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe = re.sub(r'[^\w\-_\.]', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    
    # Truncate if too long
    if len(safe) > 200:
        name, ext = os.path.splitext(safe)
        safe = name[:190] + ext
    
    return safe

def _is_pdf_file(file_path):
    """Check if file is actually a PDF by examining magic bytes."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            return header.startswith(b'%PDF-')
    except:
        return False
