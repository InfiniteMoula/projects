import os
import re
import time
from urllib.parse import urlparse

import httpx

from utils import budget_middleware, io
from utils.http import HttpError, request_with_backoff
from utils.state import SequentialRunState


def run(cfg, ctx):
    http_cfg = (cfg.get("http") or {})
    seeds = http_cfg.get("seeds") or []
    if not seeds:
        return {"status": "SKIPPED", "reason": "NO_SEEDS"}

    out = os.path.join(ctx["outdir"], "http")
    io.ensure_dir(out)
    logger = ctx.get("logger")

    per_domain_rps = float(http_cfg.get("per_domain_rps", 1.0))
    max_attempts = int(http_cfg.get("max_attempts", 5))
    backoff_factor = float(http_cfg.get("backoff_factor", 0.5))
    backoff_max = float(http_cfg.get("backoff_max", 10.0))

    budgets_cfg = (cfg.get("budgets") or {})
    max_requests = budgets_cfg.get("max_http_requests", 500)
    max_bytes = budgets_cfg.get("max_http_bytes", 10_485_760)

    state_path = os.path.join(out, "http_state.json")
    state = SequentialRunState(state_path)
    state.set_metadata(total=len(seeds))

    existing_files: list[str] = []
    completed_extra = state.metadata.get("completed_extra")
    if isinstance(completed_extra, dict):
        for info in completed_extra.values():
            if isinstance(info, dict):
                stored_path = info.get("path")
                if stored_path and os.path.exists(stored_path):
                    existing_files.append(stored_path)

    files: list[str] = list(dict.fromkeys(existing_files))
    files_seen = set(files)

    request_count = int(state.metadata.get("request_count") or 0)
    total_bytes = int(state.metadata.get("total_bytes") or 0)

    budget_tracker = ctx.get("budget_tracker")
    request_tracker = ctx.get("request_tracker")

    pending = state.pending(seeds)
    if not pending and logger:
        logger.info("collect_http_static: all %d seeds already processed", len(seeds))

    with httpx.Client(
        timeout=15,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DataCollector/1.0; +https://projects.infinitemoula.fr/robots)"
        },
    ) as client:
        for url in pending:
            if budget_tracker:
                if (
                    budget_tracker.max_http_requests
                    and budget_tracker.http_requests >= budget_tracker.max_http_requests
                ):
                    raise budget_middleware.BudgetExceededError(
                        f"HTTP request budget reached ({budget_tracker.http_requests}/{budget_tracker.max_http_requests})"
                    )
                if (
                    budget_tracker.max_http_bytes
                    and budget_tracker.http_bytes >= budget_tracker.max_http_bytes
                ):
                    raise budget_middleware.BudgetExceededError(
                        f"HTTP byte budget reached ({budget_tracker.http_bytes}/{budget_tracker.max_http_bytes})"
                    )
            else:
                if max_requests and request_count >= max_requests:
                    break
                if max_bytes and total_bytes >= max_bytes:
                    break

            if per_domain_rps > 0:
                time.sleep(1.0 / per_domain_rps)

            state.mark_started(url)

            try:
                response = request_with_backoff(
                    client,
                    "GET",
                    url,
                    max_attempts=max_attempts,
                    backoff_factor=backoff_factor,
                    backoff_max=backoff_max,
                    request_tracker=request_tracker,
                )
            except budget_middleware.BudgetExceededError:
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error=url,
                )
                raise
            except HttpError as exc:
                if logger:
                    logger.warning("Failed to fetch %s after retries: %s", url, exc)
                state.mark_failed(url, str(exc))
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error=url,
                )
                continue

            content_length = len(response.content or b"")

            request_count += 1

            if not budget_tracker and max_bytes and total_bytes + content_length > max_bytes:
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error="MAX_BYTES",
                )
                break

            total_bytes += content_length

            if response.status_code >= 400:
                if logger:
                    logger.warning(
                        "Skipping %s due to status %s",
                        url,
                        response.status_code,
                    )
                state.mark_failed(url, f"status_{response.status_code}")
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error=url,
                )
                continue

            parsed_url = urlparse(url)
            name = f"{parsed_url.netloc}_{parsed_url.path.strip('/').replace('/', '_')}"
            name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:120] + ".html"

            filepath = os.path.join(out, name)
            io.write_text(filepath, response.text)
            if filepath not in files_seen:
                files.append(filepath)
                files_seen.add(filepath)

            state.mark_completed(
                url,
                extra={
                    "path": filepath,
                    "status": response.status_code,
                    "bytes": content_length,
                },
            )
            state.set_metadata(
                request_count=request_count,
                total_bytes=total_bytes,
                last_success=url,
            )

    state.set_metadata(request_count=request_count, total_bytes=total_bytes)

    stats = {
        "requests": request_count,
        "bytes": total_bytes,
        "files": len(files),
        "state": state.stats(),
    }

    return {
        "status": "OK",
        "files": files,
        "stats": stats,
    }
