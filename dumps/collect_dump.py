
from pathlib import Path

from utils import io
from utils.http import stream_download


def run(cfg, ctx):
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir")) / "dumps"
    io.ensure_dir(outdir)
    src = ctx.get("input")
    if not src:
        return {"status": "SKIPPED", "reason": "NO_INPUT"}

    # Handle HTTP sources by downloading them
    if isinstance(src, str) and src.startswith("http"):
        # Extract filename from URL or use a default name
        url_path = Path(src.split('?')[0])  # Remove query parameters
        filename = url_path.name if url_path.name else "downloaded_data"
        if not filename.endswith(('.csv', '.parquet', '.json', '.txt')):
            filename += '.csv'  # Default extension
        
        dest_path = outdir / filename
        try:
            downloaded_path = stream_download(src, dest_path)
            file_size = downloaded_path.stat().st_size
            return {
                "source_url": src,
                "destination_file": str(downloaded_path),
                "file_size_bytes": file_size,
                "status": "OK"
            }
        except Exception as e:
            return {"status": "ERROR", "error": f"Download failed: {e}", "source_url": src}

    # Handle local file sources by copying them
    source_path = Path(src)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Copy the source file to the dumps directory
    dest_path = outdir / source_path.name
    try:
        import shutil
        shutil.copy2(source_path, dest_path)
        file_size = dest_path.stat().st_size
        return {
            "source_file": str(source_path),
            "destination_file": str(dest_path),
            "file_size_bytes": file_size,
            "status": "OK"
        }
    except Exception as e:
        return {"status": "ERROR", "error": f"File copy failed: {e}", "source_file": str(source_path)}
