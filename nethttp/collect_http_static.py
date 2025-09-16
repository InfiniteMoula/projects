import os, re, httpx
import time
from urllib.parse import urlparse, urljoin
from utils import io

def run(cfg, ctx):
    seeds = (cfg.get("http") or {}).get("seeds") or []
    if not seeds: 
        return {"status": "SKIPPED", "reason": "NO_SEEDS"}
    
    out = os.path.join(ctx["outdir"], "http")
    io.ensure_dir(out)
    files = []
    
    # Configuration parameters
    per_domain_rps = (cfg.get("http") or {}).get("per_domain_rps", 1.0)
    max_requests = (cfg.get("budgets") or {}).get("max_http_requests", 500)
    max_bytes = (cfg.get("budgets") or {}).get("max_http_bytes", 10485760)  # 10MB
    
    request_count = 0
    total_bytes = 0
    
    with httpx.Client(
        timeout=15, 
        follow_redirects=True,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; DataCollector/1.0; +https://projects.infinitemoula.fr/robots)'
        }
    ) as client:
        for url in seeds:
            if request_count >= max_requests:
                break
                
            try:
                # Respect rate limiting
                if per_domain_rps > 0:
                    time.sleep(1.0 / per_domain_rps)
                
                response = client.get(url)
                request_count += 1
                
                if response.status_code >= 400:
                    continue
                
                content_length = len(response.content)
                if total_bytes + content_length > max_bytes:
                    break
                    
                total_bytes += content_length
                
                # Generate safe filename
                parsed_url = urlparse(url)
                name = f"{parsed_url.netloc}_{parsed_url.path.strip('/').replace('/', '_')}"
                name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:120] + ".html"
                
                filepath = os.path.join(out, name)
                io.write_text(filepath, response.text)
                files.append(filepath)
                
            except Exception as e:
                # Log error but continue
                if ctx.get("logger"):
                    ctx["logger"].warning(f"Failed to fetch {url}: {e}")
                continue
    
    return {
        "status": "OK", 
        "files": files,
        "stats": {
            "requests": request_count,
            "bytes": total_bytes,
            "files": len(files)
        }
    }
