import os
import json
import time
from urllib.parse import urlparse
from utils import io

def run(cfg, ctx):
    """
    Headless browser collection for dynamic content.
    For now, this is a simplified implementation that processes static HTML files
    and simulates JavaScript rendering by extracting additional content.
    A full implementation would use playwright/selenium.
    """
    
    # Check if we have static HTML files to process
    http_dir = os.path.join(ctx["outdir"], "http")
    if not os.path.isdir(http_dir):
        return {"status": "SKIPPED", "reason": "NO_HTTP_FILES"}
    
    out = os.path.join(ctx["outdir"], "headless")
    io.ensure_dir(out)
    files = []
    
    # Get configuration
    max_pages = (cfg.get("headless") or {}).get("max_pages", 10)
    wait_time = (cfg.get("headless") or {}).get("wait_seconds", 2)
    
    processed = 0
    
    # Process HTML files from static collection
    for filename in os.listdir(http_dir):
        if not filename.endswith('.html'):
            continue
            
        if processed >= max_pages:
            break
            
        try:
            html_path = os.path.join(http_dir, filename)
            html_content = io.read_text(html_path)
            
            # Simulate headless browser processing
            enhanced_content = _simulate_js_rendering(html_content)
            
            # Save enhanced content
            output_name = filename.replace('.html', '_headless.html')
            output_path = os.path.join(out, output_name)
            io.write_text(output_path, enhanced_content)
            files.append(output_path)
            
            processed += 1
            
            # Simulate processing time
            time.sleep(0.1)
            
        except Exception as e:
            if ctx.get("logger"):
                ctx["logger"].warning(f"Failed to process {filename}: {e}")
            continue
    
    return {
        "status": "OK",
        "files": files,
        "stats": {
            "processed": processed,
            "enhanced_files": len(files)
        }
    }

def _simulate_js_rendering(html_content):
    """
    Simulate JavaScript rendering by:
    - Extracting data-* attributes that might be populated by JS
    - Looking for JSON data in script tags
    - Adding structured data markers
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for script tags with JSON data
    json_scripts = soup.find_all('script', type='application/json')
    for script in json_scripts:
        if script.string:
            try:
                data = json.loads(script.string)
                # Add this data as a comment for parsing later
                comment = soup.new_string(f"<!-- HEADLESS_DATA: {json.dumps(data)} -->", soup.Comment)
                script.insert_after(comment)
            except json.JSONDecodeError:
                pass
    
    # Look for elements with data-* attributes that might contain contact info
    for element in soup.find_all(attrs={"data-email": True}):
        email = element.get("data-email")
        if email and "@" in email:
            # Make this email visible in text content
            element.string = f"Email: {email}"
    
    for element in soup.find_all(attrs={"data-phone": True}):
        phone = element.get("data-phone")
        if phone:
            element.string = f"Phone: {phone}"
    
    # Add metadata about headless processing
    meta_tag = soup.new_tag("meta", attrs={
        "name": "headless-processed",
        "content": "true"
    })
    if soup.head:
        soup.head.append(meta_tag)
    
    return str(soup)
