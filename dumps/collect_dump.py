
from pathlib import Path

from utils import io
from utils.http import stream_download


def run(cfg, ctx):
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir")) / "dumps"
    io.ensure_dir(outdir)
    src = ctx.get("input")
    if not src:
        return {"status": "SKIPPED", "reason": "NO_INPUT"}

    # Skip actual file copying to prevent filling up disk space with raw data dumps
    # Return metadata about the source file instead of copying it
    if isinstance(src, str) and src.startswith("http"):
        # For HTTP sources, just return the original URL without downloading
        return {"source_url": src, "status": "SKIPPED", "reason": "RAW_DUMP_DISABLED"}

    source_path = Path(src)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Return source file info without copying to save disk space
    file_size = source_path.stat().st_size
    return {
        "source_file": str(source_path), 
        "file_size_bytes": file_size,
        "status": "SKIPPED", 
        "reason": "RAW_DUMP_DISABLED"
    }
