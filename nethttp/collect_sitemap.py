import os

import httpx

from utils import budget_middleware, io
from utils.http import HttpError, request_with_backoff
from utils.state import SequentialRunState
from config.budget_config import get_budget_thresholds

BUDGET_DEFAULTS = get_budget_thresholds()


def _resolve_budget(value, default):
    if value is None:
        return int(default or 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default or 0)


def run(cfg, ctx):
    sitemap_cfg = (cfg.get("sitemap") or {})
    domains = sitemap_cfg.get("domains") or []
    if not domains:
        return {"status": "SKIPPED", "reason": "NO_DOMAINS"}

    out = os.path.join(ctx["outdir"], "sitemaps")
    io.ensure_dir(out)
    logger = ctx.get("logger")

    max_attempts = int(sitemap_cfg.get("max_attempts", 5))
    backoff_factor = float(sitemap_cfg.get("backoff_factor", 0.5))
    backoff_max = float(sitemap_cfg.get("backoff_max", 10.0))

    budgets_cfg = (cfg.get("budgets") or {})
    if not isinstance(budgets_cfg, dict):
        budgets_cfg = {}
    max_requests = _resolve_budget(
        budgets_cfg.get("max_http_requests"), BUDGET_DEFAULTS.max_http_requests
    )
    max_bytes = _resolve_budget(
        budgets_cfg.get("max_http_bytes"), BUDGET_DEFAULTS.max_http_bytes
    )

    state_path = os.path.join(out, "sitemap_state.json")
    state = SequentialRunState(state_path)
    state.set_metadata(total=len(domains))

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

    pending = state.pending(domains)
    if not pending and logger:
        logger.info("collect_sitemap: all %d domains already processed", len(domains))

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        for domain in pending:
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

            url = f"https://{domain}/sitemap.xml"
            state.mark_started(domain)
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
                    last_error=domain,
                )
                raise
            except HttpError as exc:
                if logger:
                    logger.warning("Failed to fetch sitemap %s after retries: %s", url, exc)
                state.mark_failed(domain, str(exc))
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error=domain,
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
                        "Skipping sitemap %s due to status %s",
                        url,
                        response.status_code,
                    )
                state.mark_failed(domain, f"status_{response.status_code}")
                state.set_metadata(
                    request_count=request_count,
                    total_bytes=total_bytes,
                    last_error=domain,
                )
                continue

            filepath = os.path.join(out, domain.replace(".", "_") + "_sitemap.xml")
            io.write_text(filepath, response.text)
            if filepath not in files_seen:
                files.append(filepath)
                files_seen.add(filepath)

            state.mark_completed(
                domain,
                extra={
                    "path": filepath,
                    "status": response.status_code,
                    "bytes": content_length,
                },
            )
            state.set_metadata(
                request_count=request_count,
                total_bytes=total_bytes,
                last_success=domain,
            )

    state.set_metadata(request_count=request_count, total_bytes=total_bytes)

    stats = {
        "requests": request_count,
        "bytes": total_bytes,
        "files": len(files),
        "state": state.stats(),
    }

    return {"status": "OK", "files": files, "stats": stats}
