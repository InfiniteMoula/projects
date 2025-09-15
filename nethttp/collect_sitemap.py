import os, httpx
from utils import io

def run(cfg, ctx):
    domains=(cfg.get("sitemap") or {}).get("domains") or []
    if not domains: return {"status":"SKIPPED","reason":"NO_DOMAINS"}
    out=os.path.join(ctx["outdir"],"sitemaps"); io.ensure_dir(out); files=[]
    with httpx.Client(timeout=15, follow_redirects=True) as c:
        for d in domains:
            try:
                url=f"https://{d}/sitemap.xml"
                r=c.get(url)
                if r.status_code<400:
                    p=os.path.join(out,d.replace(".","_")+"_sitemap.xml")
                    io.write_text(p,r.text); files.append(p)
            except Exception:
                pass
    return {"status":"OK","files":files}
