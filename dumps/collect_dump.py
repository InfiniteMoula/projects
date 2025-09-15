
from pathlib import Path

from utils import io
from utils.http import stream_download


def run(cfg, ctx):
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir")) / "dumps"
    io.ensure_dir(outdir)
    src = ctx.get("input")
    if not src:
        return {"status": "SKIPPED", "reason": "NO_INPUT"}

    timeout = float((cfg.get("dumps") or {}).get("timeout", 60))
    if isinstance(src, str) and src.startswith("http"):
        dest = outdir / Path(src).name
        path = stream_download(src, dest, timeout=timeout, resume=True)
        return {"file": str(path)}

    source_path = Path(src)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    dest = outdir / source_path.name
    io.atomic_write_iter(dest, iter(lambda: source_path.open("rb").read(1 << 20), b""))
    return {"file": str(dest)}
