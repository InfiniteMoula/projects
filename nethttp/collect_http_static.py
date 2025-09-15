import os, re, httpx
from utils import io

def run(cfg, ctx):
    seeds=(cfg.get("http") or {}).get("seeds") or []
    if not seeds: return {"status":"SKIPPED","reason":"NO_SEEDS"}
    out=os.path.join(ctx["outdir"],"http"); io.ensure_dir(out); files=[]
    with httpx.Client(timeout=15, follow_redirects=True) as c:
        for url in seeds:
            try:
                r=c.get(url)
                if r.status_code>=400: continue
                name=re.sub(r"[^a-zA-Z0-9]+","_",url)[:120]+".html"
                p=os.path.join(out,name); io.write_text(p,r.text); files.append(p)
            except Exception:
                pass
    return {"status":"OK","files":files}
