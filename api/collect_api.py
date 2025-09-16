
from pathlib import Path

from utils import io
from utils.http import get_json


def run(cfg, ctx):
    """Call public APIs defined in job['api']['endpoints'] with rate limit. Dry-run returns empty."""
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir")) / "api"
    io.ensure_dir(outdir)
    endpoints = (cfg.get("api") or {}).get("endpoints") or []
    if ctx.get("dry_run") or not endpoints:
        path = outdir / "empty.json"
        io.write_json(path, {"note": "dry_run_or_no_endpoints"})
        return {"file": str(path)}
    files = []
    timeout = float((cfg.get("api") or {}).get("timeout", 20))
    request_tracker = ctx.get("request_tracker")
    for ep in endpoints:
        data = get_json(ep["url"], params=ep.get("params") or {}, timeout=timeout, request_tracker=request_tracker)
        path = outdir / f"{ep['name']}.json"
        io.write_json(path, data)
        files.append(str(path))
    return {"files": files}
