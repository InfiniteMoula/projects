# FILE: parse/parse_jsonld.py
import os, glob, json
from bs4 import BeautifulSoup
from utils import io
def run(cfg, ctx):
    html_dir=os.path.join(ctx["outdir"],"http")
    if not os.path.isdir(html_dir): return {"status":"SKIPPED","reason":"NO_HTML"}
    out=os.path.join(ctx["outdir"],"jsonld"); io.ensure_dir(out); n=0
    for p in glob.glob(os.path.join(html_dir,"*.html")):
        s=open(p,encoding="utf-8",errors="ignore").read()
        for t in BeautifulSoup(s,"lxml").select('script[type="application/ld+json"]'):
            try: json.loads(t.string or ""); n+=1
            except: pass
    io.write_json(os.path.join(out,"extracted.json"), {"count":n})
    return {"file":os.path.join(out,"extracted.json"),"count":n}
