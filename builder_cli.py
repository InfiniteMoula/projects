import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from pathlib import Path

import psutil
import yaml
from jsonschema import ValidationError as SchemaValidationError, validate as js_validate
from config.enrichment_config import load_enrichment_config

import create_job
from metrics.collector import get_metrics
from config.budget_config import get_budget_thresholds

from utils import budget_middleware, config, io, pipeline
from utils.adaptive_controller import AdaptiveController, AdaptiveState
from utils.ua import load_user_agent_pool
from utils.directories import initialize_pipeline_directories

try:
    from monitoring import prometheus_exporter
except ImportError:  # pragma: no cover - optional dependency
    prometheus_exporter = None

LOGGER = logging.getLogger(__name__)
METRICS = get_metrics()
BUDGET_DEFAULTS = get_budget_thresholds()
TIME_BUDGET_DEFAULT = int(BUDGET_DEFAULTS.time_budget_min or 0)
RAM_BUDGET_DEFAULT = int(BUDGET_DEFAULTS.ram_mb or 0)


def _deep_update(target: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping):
            current = target.get(key)
            if isinstance(current, dict):
                _deep_update(current, value)
            else:
                target[key] = copy.deepcopy(dict(value))
        else:
            target[key] = copy.deepcopy(value)
    return target


STEP_REGISTRY = {
    "dumps.collect": "dumps.collect_dump:run",
    "api.collect": "api.collect_api:run",
    "http.static": "nethttp.collect_http_static:run",
    "http.sitemap": "nethttp.collect_sitemap:run",
    "http.serp": "nethttp.collect_serp:run",
    "crawl.site": "nethttp.crawl_site:run",
    "crawl.site_async": "nethttp.crawl_site_async:run",
    "headless.collect": "headless.collect_headless:run",
    "headless.collect_fallback": "headless.collect_headless_fallback:run",
    "feeds.collect": "feeds.collect_rss:run",
    "pdf.collect": "pdf.collect_pdf:run",
    "parse.html": "parse.parse_html:run",
    "parse.jsonld": "parse.parse_jsonld:run",
    "parse.pdf": "parse.parse_pdf:run",
    "parse.contacts": "parse.parse_contacts:run",
    "parse.contacts.initial": "parse.parse_contacts:run",
    "parse.contacts.final": "parse.parse_contacts:run",
    "parse.contacts.ai": "parse.parse_contacts_ai:run",
    "normalize.standardize": "normalize.standardize:run",
    "enrich.domain": "enrich.domain_discovery:run",
    "enrich.site": "enrich.site_probe:run",
    "enrich.dns": "enrich.dns_checks:run",
    "enrich.email": "enrich.email_heuristics:run",
    "enrich.phone": "enrich.phone_checks:run",
    "enrich.address": "enrich.address_search:run",
    "enrich.google_maps": "enrich.google_maps_search:run",
    "enrich.domains": "enrich.enrich_domains:run",
    "enrich.contacts": "enrich.enrich_contacts:run",
    "enrich.linkedin": "enrich.enrich_linkedin:run",
    "enrich.linkedin_clearbit_lite": "enrich.linkedin_clearbit_lite:run",
    "correlation.checks": "correlation.correlation:run",
    "scraper.maps": "scraper.maps_scraper:run",
    "quality.checks": "quality.checks:run",
    "quality.dedupe": "quality.dedupe:run",
    "quality.score": "quality.score:run",
    "quality.enrich_score": "quality.enrich_score:run",
    "quality.clean_contacts": "quality.clean_contacts:run",
    "quality.clean_contacts.initial": "quality.clean_contacts:run",
    "quality.clean_contacts.final": "quality.clean_contacts:run",
    "monitor.scraper": "monitor.monitor_scraper:run",
    "package.export": "package.exporter:run",
    "metrics.export": "metrics.exporter:run",
    "finalize.premium_dataset": "utils.filters:run_finalize_premium_dataset",
    "top500.load_dataset": "pipeline.top_500_premium:load_enriched_dataset",
    "top500.filter_sellable": "pipeline.top_500_premium:filter_sellable",
    "top500.ensure_score": "pipeline.top_500_premium:ensure_score",
    "top500.sort_score": "pipeline.top_500_premium:sort_by_score",
    "top500.limit": "pipeline.top_500_premium:select_head",
    "top500.reorder": "pipeline.top_500_premium:reorder_for_export",
    "top500.export_csv": "pipeline.top_500_premium:export_csv",
}

STEP_DEPENDENCIES = {
    "headless.collect": {"dumps.collect"},
    "headless.collect_fallback": {"quality.clean_contacts.initial"},
    "feeds.collect": {"dumps.collect"},
    "pdf.collect": {"dumps.collect"},
    "parse.html": {"headless.collect"},
    "parse.jsonld": {"feeds.collect"},
    "parse.pdf": {"pdf.collect"},
    "parse.contacts": {"crawl.site"},
    "parse.contacts.initial": {"crawl.site_async"},
    "parse.contacts.final": {"headless.collect_fallback"},
    "parse.contacts.ai": {"quality.clean_contacts.initial"},
    "http.serp": {"enrich.address", "normalize.standardize"},
    "crawl.site": {"http.serp"},
    "crawl.site_async": {"http.serp"},
    "normalize.standardize": {
        "dumps.collect",
        "api.collect",
        "parse.html",
        "parse.jsonld",
        "parse.pdf",
    },
    "enrich.domain": {"enrich.google_maps"},
    "enrich.site": {"enrich.google_maps"},
    "enrich.dns": {"enrich.google_maps"},
    "enrich.email": {"enrich.google_maps"},
    "enrich.phone": {"enrich.google_maps"},
    "enrich.address": {"normalize.standardize"},
    "enrich.domains": {"normalize.standardize", "enrich.address"},
    "enrich.contacts": {"enrich.domains"},
    "enrich.linkedin": {"normalize.standardize"},
    "enrich.linkedin_clearbit_lite": {"normalize.standardize"},
    "correlation.checks": {"enrich.contacts"},
    "enrich.google_maps": {"enrich.address"},
    "scraper.maps": {"normalize.standardize", "enrich.address"},
    "quality.checks": {"normalize.standardize"},
    "quality.dedupe": {"enrich.email", "normalize.standardize"},
    "quality.score": {"normalize.standardize"},
    "quality.enrich_score": {"parse.contacts", "quality.checks"},
    "quality.clean_contacts": {"parse.contacts"},
    "quality.clean_contacts.initial": {"parse.contacts.initial"},
    "quality.clean_contacts.final": {"parse.contacts.final"},
    "monitor.scraper": {"quality.clean_contacts.final"},
    "package.export": {"quality.score"},
    "metrics.export": {"quality.score"},
    "finalize.premium_dataset": {"package.export"},
    "top500.filter_sellable": {"top500.load_dataset"},
    "top500.ensure_score": {"top500.filter_sellable"},
    "top500.sort_score": {"top500.ensure_score"},
    "top500.limit": {"top500.sort_score"},
    "top500.reorder": {"top500.limit"},
    "top500.export_csv": {"top500.reorder"},
}

ENRICHMENT_STEP_FLAGS = {
    "enrich.domains": "use_domains",
    "enrich.contacts": "use_contacts",
    "enrich.linkedin": "use_linkedin",
    "enrich.linkedin_clearbit_lite": "use_linkedin",
    "correlation.checks": "use_correlation",
    "parse.contacts.ai": "use_contacts_ai",
    "metrics.export": "use_metrics_export",
}

PROFILES = {
    "quick": [
        "dumps.collect",
        "api.collect",
        "normalize.standardize",
        "quality.checks",
        "quality.score",
        "package.export",
        "finalize.premium_dataset",
    ],
    "standard": [
        "dumps.collect",
        "api.collect",
        "feeds.collect",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.address",
        "enrich.google_maps",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "enrich.linkedin_clearbit_lite",
        "quality.checks",
        "quality.score",
        "package.export",
        "finalize.premium_dataset",
    ],
    "standard_nocapital": [
        "dumps.collect",
        "api.collect",
        "normalize.standardize",
        "enrich.address",
        "enrich.domains",
        "enrich.contacts",
        "correlation.checks",
        "enrich.linkedin",
        "quality.checks",
        "quality.score",
        "package.export",
    ],
    "standard_nocapital_v2": [
        "dumps.collect",
        "api.collect",
        "normalize.standardize",
        "enrich.address",
        "enrich.domains",
        "enrich.contacts",
        "correlation.checks",
        "quality.checks",
        "quality.score",
        "metrics.export",
        "package.export",
    ],
    "hybrid": [
        "http.serp",
        "crawl.site_async",
        "parse.contacts.initial",
        "quality.clean_contacts.initial",
        "headless.collect_fallback",
        "parse.contacts.final",
        "quality.clean_contacts.final",
        "monitor.scraper",
        "package.export",
        "finalize.premium_dataset",
    ],
    "deep": [
        "dumps.collect",
        "api.collect",
        "headless.collect",
        "feeds.collect",
        "pdf.collect",
        "parse.pdf",
        "parse.html",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.address",
        "enrich.google_maps",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "enrich.linkedin_clearbit_lite",
        "quality.checks",
        "quality.score",
        "package.export",
        "finalize.premium_dataset",
    ],
    "internal": [
        "dumps.collect",
        "api.collect",
        "feeds.collect",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.address",
        "http.serp",
        "crawl.site",
        "parse.contacts",
        "scraper.maps",
        "quality.checks",
        "quality.enrich_score",
        "quality.score",
        "package.export",
        "finalize.premium_dataset",
    ],
    "top_500_premium": [
        "top500.load_dataset",
        "top500.filter_sellable",
        "top500.ensure_score",
        "top500.sort_score",
        "top500.limit",
        "top500.reorder",
        "top500.export_csv",
    ],
}


PROFILE_FEATURE_OVERRIDES = {
    "standard_nocapital_v2": {
        "flags": {
            "use_correlation": True,
            "use_metrics_export": True,
        },
        "config": {
            "cache": {
                "enabled": True,
                "backend": "sqlite",
            },
            "circuit_breaker": {
                "enabled": True,
            },
            "embeddings": {
                "enabled": False,
            },
            "ai": {
                "enabled": False,
                "contacts": False,
            },
        },
    },
}


STEP_OUTPUT_HINTS = {
    "http.serp": ["serp/serp_results.parquet"],
    "crawl.site": ["crawl/pages.parquet"],
    "crawl.site_async": ["crawl/pages.parquet"],
    "parse.contacts": ["contacts/contacts.parquet"],
    "parse.contacts.initial": ["contacts/contacts.parquet"],
    "parse.contacts.final": ["contacts/contacts.parquet"],
    "parse.contacts.ai": ["contacts/contacts_ai.parquet"],
    "quality.clean_contacts": ["contacts/contacts_clean.parquet", "contacts/no_contact.csv"],
    "quality.clean_contacts.initial": ["contacts/contacts_clean.parquet", "contacts/no_contact.csv"],
    "quality.clean_contacts.final": ["contacts/contacts_clean.parquet"],
    "headless.collect_fallback": ["headless/pages_dynamic.parquet"],
    "monitor.scraper": ["metrics/summary.json", "metrics/scraper_stats.csv"],
    "finalize.premium_dataset": ["dataset.csv", "dataset.parquet"],
    "metrics.export": ["reports/report_metrics.json"],
    "enrich.linkedin_clearbit_lite": ["linkedin_clearbit_lite.parquet"],
}


def _resolve_step_outputs(step_name: str, context: dict) -> List[Path]:
    hints = STEP_OUTPUT_HINTS.get(step_name, [])
    if not hints:
        return []
    base = Path(context.get("outdir") or ".")
    return [base / hint for hint in hints]


def _register_resume_outputs(ctx: dict, status: dict) -> None:
    outputs = status.get("outputs")
    if not outputs:
        return
    fresh = ctx.setdefault("_fresh_outputs", set())
    for output in outputs:
        if output:
            fresh.add(str(output))


def _is_enrichment_step_enabled(step_name: str, context: dict) -> bool:
    flag_name = ENRICHMENT_STEP_FLAGS.get(step_name)
    if not flag_name:
        return True
    flags = context.get("enrichment_flags") or {}
    return bool(flags.get(flag_name, True))


def _mark_enrichment_step_disabled(step_name: str, context: dict) -> dict:
    logger = context.get("logger") or pipeline.get_logger()
    log_lock = context.get("log_lock")
    status = {
        "step": step_name,
        "status": "SKIPPED",
        "reason": "disabled",
        "duration_s": 0,
    }
    pipeline.log_step_event(logger, step_name, "disabled", status="SKIPPED", reason="disabled")
    logs_path = context.get("logs")
    if logs_path:
        with (log_lock if log_lock else nullcontext()):
            io.log_json(logs_path, status)
    return status


def load_job(job_path: os.PathLike[str] | str) -> dict:
    job_file = Path(job_path).expanduser().resolve()
    try:
        job = yaml.safe_load(io.read_text(job_file))
    except io.IoError as exc:
        raise RuntimeError(f"Unable to read job file: {job_file}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in job file {job_file}: {exc}") from exc

    schema_path = job_file.with_name("job_schema.json")
    if schema_path.exists():
        try:
            schema = json.loads(io.read_text(schema_path))
        except io.IoError as exc:
            raise RuntimeError(f"Unable to read schema file: {schema_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON schema in {schema_path}: {exc}") from exc
        try:
            js_validate(job, schema)
        except SchemaValidationError as exc:
            raise ValueError(f"Job file {job_file} does not match schema: {exc.message}") from exc

    job["_job_path"] = str(job_file)
    return job


def _check_ram(ctx):
    budget = int(ctx.get("max_ram_mb", 0) or 0)  # 0 => desactive
    if budget <= 0:
        return
    used_mb = psutil.Process().memory_info().rss // (1024 * 1024)
    if used_mb > budget:
        raise RuntimeError(f"BUDGET_REACHED_RAM: {used_mb} MiB > {budget} MiB")


def _run_step(step_name, cfg, context):
    logger = context.get("logger") or pipeline.get_logger()
    budget_tracker = context.get("budget_tracker")
    verbose = context.get("verbose", False)
    debug = context.get("debug", False)
    outputs = _resolve_step_outputs(step_name, context)
    fresh_outputs = context.setdefault("_fresh_outputs", set())
    log_lock = context.get("log_lock")

    if not _is_enrichment_step_enabled(step_name, context):
        return _mark_enrichment_step_disabled(step_name, context)

    if context.get("resume"):
        if outputs and all(path.exists() for path in outputs):
            if not any(str(path) in fresh_outputs for path in outputs):
                status = {
                    "step": step_name,
                    "status": "SKIPPED",
                    "reason": "resume",
                    "duration_s": 0,
                    "outputs": [str(p) for p in outputs],
                }
                pipeline.log_step_event(
                    logger,
                    step_name,
                    "resume_skip",
                    status="SKIPPED",
                    reason="resume",
                )
                with (log_lock if log_lock else nullcontext()):
                    io.log_json(context["logs"], status)
                return status
    
    # Debug info: Step configuration and context
    if debug:
        logger.info(f"[DEBUG] Starting step '{step_name}'")
        logger.info(f"[DEBUG] Step dependencies: {STEP_DEPENDENCIES.get(step_name, [])}")
        if verbose:
            logger.debug(f"[VERBOSE] Step configuration keys: {list(cfg.keys()) if cfg else 'None'}")
            logger.debug(f"[VERBOSE] Context keys: {list(context.keys())}")
    
    # Check time budget before starting step
    if budget_tracker:
        try:
            budget_tracker.check_time_budget()
            if debug:
                logger.info(f"[DEBUG] Budget check passed for step '{step_name}'")
        except budget_middleware.BudgetExceededError as exc:
            if debug:
                logger.info(f"[DEBUG] Budget exceeded before step '{step_name}': {exc}")
            status = {
                "step": step_name,
                "status": "BUDGET_EXCEEDED",
                "error": str(exc),
                "duration_s": 0,
            }
            pipeline.log_step_event(logger, step_name, "budget_exceeded", status="BUDGET_EXCEEDED", error=str(exc))
            with (log_lock if log_lock else nullcontext()):
                io.log_json(context["logs"], status)
            raise RuntimeError(f"step {step_name} budget exceeded: {exc}")
    
    fn = pipeline.resolve_step(step_name, STEP_REGISTRY)
    if debug:
        logger.info(f"[DEBUG] Resolved step function: {fn.__module__}.{fn.__name__}")
    
    pipeline.log_step_event(logger, step_name, "start")
    started = time.time()
    try:
        _check_ram(context)
        if debug:
            ram_info = psutil.Process().memory_info()
            logger.info(f"[DEBUG] RAM usage before step: {ram_info.rss // (1024 * 1024)} MB")
        
        # Add budget tracker to context for steps to use
        step_context = context.copy()
        if budget_tracker:
            step_context["budget_tracker"] = budget_tracker
            step_context["request_tracker"] = lambda size: budget_tracker.track_http_request(size)
        
        if debug:
            logger.info(f"[DEBUG] Executing step '{step_name}'...")
        
        out = fn(cfg, step_context) or {}

        if debug:
            logger.info(f"[DEBUG] Step '{step_name}' completed with status: {out.get('status', 'OK')}")
            if verbose and out:
                logger.debug(f"[VERBOSE] Step output: {out}")
        
        _check_ram(context)
        if debug:
            ram_info = psutil.Process().memory_info()
            logger.info(f"[DEBUG] RAM usage after step: {ram_info.rss // (1024 * 1024)} MB")
        
        # Check budgets after step completion
        if budget_tracker:
            budget_tracker.check_time_budget()
            if debug:
                budget_stats = budget_tracker.get_current_stats()
                logger.info(f"[DEBUG] Budget stats after step: {budget_stats}")
            
        status = {
            "step": step_name,
            "status": out.get("status") or "OK",
            "out": out,
            "duration_s": round(time.time() - started, 3),
        }
        if outputs:
            status["outputs"] = [str(p) for p in outputs]
        
        # Add budget stats to status
        if budget_tracker:
            status["budget_stats"] = budget_tracker.get_current_stats()
            
    except budget_middleware.BudgetExceededError as exc:
        if debug:
            logger.info(f"[DEBUG] Budget exceeded during step '{step_name}': {exc}")
        status = {
            "step": step_name,
            "status": "BUDGET_EXCEEDED", 
            "error": str(exc),
            "duration_s": round(time.time() - started, 3),
        }
        if budget_tracker:
            status["budget_stats"] = budget_tracker.get_current_stats()
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        logger.exception("Step %s failed", step_name)
        status = {
            "step": step_name,
            "status": "FAIL",
            "error": str(exc),
            "trace": traceback.format_exc(),
            "duration_s": round(time.time() - started, 3),
        }
        if budget_tracker:
            status["budget_stats"] = budget_tracker.get_current_stats()
    
    extra = {}
    if status.get("error"):
        extra["error"] = status["error"]
    pipeline.log_step_event(
        logger,
        step_name,
        "end",
        status=status["status"],
        duration=status["duration_s"],
        **extra,
    )
    with (log_lock if log_lock else nullcontext()):
        io.log_json(context["logs"], status)
    if status["status"] == "OK" and outputs:
        for path in outputs:
            fresh_outputs.add(str(path))
    if verbose:
        logger.debug(f"[VERBOSE] Complete step result for {step_name}: {json.dumps(status, ensure_ascii=False)}")
    elif debug and status.get("error"):
        logger.info(f"[DEBUG] Step '{step_name}' result: {status['status']} in {status['duration_s']}s")

    prom_ctx = context.get("_prometheus")
    if isinstance(prom_ctx, dict) and prometheus_exporter:
        try:
            prometheus_exporter.observe_step(step_name, status["status"], status["duration_s"])
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Prometheus export failed for step %s", step_name, exc_info=True)

    if status["status"] not in ("OK", "SKIPPED", "WARN"):
        raise RuntimeError(f"step {step_name} failed")
    return status


def _update_adaptive_context(ctx: dict, state: AdaptiveState) -> None:
    enrichment_cfg = ctx.get("enrichment_config")
    if not isinstance(enrichment_cfg, dict):
        return

    changed = state.concurrency_changed or state.chunk_size_changed

    domains_cfg = enrichment_cfg.setdefault("domains", {})
    if isinstance(domains_cfg, dict):
        domains_cfg["chunk_size"] = state.chunk_size
        http_cfg = domains_cfg.setdefault("http_client", {})
        if isinstance(http_cfg, dict):
            http_cfg["max_concurrent_requests"] = state.concurrency

    for section in ("contacts", "linkedin"):
        section_cfg = enrichment_cfg.get(section)
        if not isinstance(section_cfg, dict):
            continue
        http_cfg = section_cfg.setdefault("http_client", {})
        if isinstance(http_cfg, dict):
            http_cfg["max_concurrent_requests"] = state.concurrency

    if changed:
        logger = ctx.get("logger") or pipeline.get_logger()
        logger.info(
            "Adaptive controller updated targets: concurrency=%s, chunk_size=%s",
            state.concurrency,
            state.chunk_size,
        )


def _apply_adaptive_feedback(ctx: dict, batch_results: Sequence[dict]) -> None:
    controller: Optional[AdaptiveController] = ctx.get("adaptive_controller")
    if not controller or not batch_results:
        return

    total = len(batch_results)
    failures = sum(
        1 for result in batch_results if result.get("status") not in {"OK", "SKIPPED", "WARN"}
    )

    total_requests = 0.0
    total_duration = 0.0
    for result in batch_results:
        budget_stats = result.get("budget_stats") or {}
        total_requests += float(budget_stats.get("http_requests", 0.0) or 0.0)
        total_duration += float(result.get("duration_s", 0.0) or 0.0)

    req_per_min = None
    if total_duration > 0 and total_requests > 0:
        req_per_min = (total_requests / total_duration) * 60.0

    ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    state = controller.observe(
        error_rate=failures / total if total else 0.0,
        req_per_min=req_per_min,
        ram_used=ram_gb,
    )

    _update_adaptive_context(ctx, state)


def build_context(args, job):
    run_id = args.run_id or uuid.uuid4().hex[:12]
    outdir_path = Path(args.out).expanduser().resolve()
    dir_map = dict(initialize_pipeline_directories(outdir_path))
    logs_dir = dir_map.get("logs")
    if logs_dir is None:
        logs_dir = io.ensure_dir(outdir_path / "logs")
        dir_map["logs"] = logs_dir

    input_path = None
    if args.input:
        input_candidate = Path(args.input).expanduser().resolve()
        if not input_candidate.exists():
            raise FileNotFoundError(f"Input path not found: {input_candidate}")
        input_path = input_candidate

    profile_name = getattr(args, "profile", None) or job.get("profile")

    ctx = {
        "run_id": run_id,
        "outdir": str(outdir_path),
        "outdir_path": outdir_path,
        "logs": str((logs_dir / f"{run_id}.json")),
        "logs_path": logs_dir / f"{run_id}.json",
        "dry_run": args.dry_run,
        "sample": args.sample,
        "lang": job.get("output", {}).get("lang", "fr"),
        "time_budget_min": args.time_budget_min,
        "workers": args.workers,
        "env": config.load_env(),
        "job": job,
        "job_path": job.get("_job_path", ""),
        "input": str(input_path) if input_path else None,
        "input_path": input_path,
        "resume": args.resume,
        "json": args.json,
        "verbose": args.verbose,
        "debug": getattr(args, 'debug', False),
        "max_ram_mb": args.max_ram_mb,
        "metrics_file": str(args.metrics_file) if getattr(args, "metrics_file", None) else None,
        "parallel": getattr(args, 'parallel', False),
        "parallel_mode": getattr(args, 'parallel_mode', 'thread'),
        "log_lock": threading.Lock(),
        "_fresh_outputs": set(),
        "profile": profile_name,
        "directories": dir_map,
    }
    ctx["serp_timeout_sec"] = float(getattr(args, "serp_timeout_sec", 8.0))
    ctx["max_pages_per_domain"] = int(getattr(args, "max_pages_per_domain", 12))
    ctx["crawl_time_budget_min"] = float(getattr(args, "crawl_time_budget_min", 60.0))
    respect = getattr(args, "respect_robots", None)
    ctx["respect_robots"] = True if respect is None else bool(respect)
    ua_path = getattr(args, "user_agent_pool", None)
    ctx["user_agent_pool_path"] = str(ua_path) if ua_path else None
    ctx["user_agent_pool"] = load_user_agent_pool(str(ua_path) if ua_path else None)
    concurrency = int(getattr(args, "concurrency", 0) or 0)
    if concurrency > 0:
        ctx["max_domains_parallel"] = concurrency
    per_domain_rps = float(getattr(args, "per_domain_rps", 0.0) or 0.0)
    if per_domain_rps > 0:
        ctx["per_host_rps"] = per_domain_rps
    
    # Initialize budget tracker if budgets are configured
    budget_tracker = budget_middleware.create_budget_tracker(job)
    if budget_tracker:
        ctx["budget_tracker"] = budget_tracker
        
    # Initialize KPI calculator if targets are configured  
    kpi_calculator = budget_middleware.create_kpi_calculator(job)
    if kpi_calculator:
        ctx["kpi_calculator"] = kpi_calculator

    prom_port = int(getattr(args, "prometheus_port", 0) or 0)
    prom_addr = getattr(args, "prometheus_address", "0.0.0.0")
    ctx["_prometheus"] = None
    if prom_port and prometheus_exporter:
        started = prometheus_exporter.start_metrics_server(prom_port, prom_addr)
        if started:
            ctx["_prometheus"] = {
                "run_id": run_id,
                "job_name": Path(args.job).stem,
                "port": prom_port,
                "address": prom_addr,
            }
        else:
            ctx["_prometheus"] = False

    cfg_path = getattr(args, "enrichment_config", None) or Path("config/enrichment.yaml")
    cfg_path = Path(cfg_path).expanduser()
    enrichment_cfg = getattr(args, "enrich_cfg", None)
    if enrichment_cfg is None:
        enrichment_cfg = load_enrichment_config(cfg_path)
    ctx["enrichment_config_path"] = str(cfg_path.resolve())
    ctx["enrich_cfg"] = enrichment_cfg
    flags = {
        "use_domains": enrichment_cfg.use_domains,
        "use_contacts": enrichment_cfg.use_contacts,
        "use_linkedin": enrichment_cfg.use_linkedin,
        "use_correlation": getattr(enrichment_cfg, "use_correlation", True),
        "use_metrics_export": getattr(enrichment_cfg, "use_metrics_export", False),
        "use_contacts_ai": enrichment_cfg.ai.contacts_enabled,
    }
    enrichment_config = enrichment_cfg.model_dump()

    profile_overrides = PROFILE_FEATURE_OVERRIDES.get(profile_name or "")
    if profile_overrides:
        flags.update(profile_overrides.get("flags", {}))
        config_overrides = profile_overrides.get("config") or {}
        if isinstance(config_overrides, Mapping):
            _deep_update(enrichment_config, config_overrides)

    for flag_name in flags:
        override = getattr(args, flag_name, None)
        if override is not None:
            flags[flag_name] = bool(override)

    for key in ("use_domains", "use_contacts", "use_linkedin", "use_correlation", "use_metrics_export"):
        enrichment_config[key] = bool(flags.get(key, False))

    ai_cfg = enrichment_config.get("ai") or {}
    if not isinstance(ai_cfg, dict):
        ai_cfg = {}
    ai_cfg["contacts"] = bool(flags.get("use_contacts_ai", False))
    if ai_cfg["contacts"]:
        ai_cfg["enabled"] = True
    else:
        ai_cfg["enabled"] = bool(ai_cfg.get("enabled", False) and flags.get("use_contacts_ai", False))
    enrichment_config["ai"] = ai_cfg
    flags["use_contacts_ai"] = bool(ai_cfg.get("enabled") and ai_cfg.get("contacts"))

    embeddings_cfg = enrichment_config.get("embeddings")
    if isinstance(embeddings_cfg, Mapping):
        domains_cfg = enrichment_config.setdefault("domains", {})
        if not isinstance(domains_cfg, dict):
            domains_cfg = {}
            enrichment_config["domains"] = domains_cfg
        domain_embeddings = domains_cfg.setdefault("embeddings", {})
        if not isinstance(domain_embeddings, dict):
            domain_embeddings = {}
            domains_cfg["embeddings"] = domain_embeddings
        _deep_update(domain_embeddings, embeddings_cfg)
    adaptive_cfg = enrichment_cfg.adaptive
    if adaptive_cfg.enabled:
        domains_cfg = enrichment_config.setdefault("domains", {})
        if isinstance(domains_cfg, dict):
            http_cfg = domains_cfg.setdefault("http_client", {})
            initial_concurrency = int(
                http_cfg.get("max_concurrent_requests") or adaptive_cfg.min_concurrency
            )
            initial_chunk = int(domains_cfg.get("chunk_size") or adaptive_cfg.min_chunk)
        else:
            domains_cfg = {}
            enrichment_config["domains"] = domains_cfg
            http_cfg = {}
            domains_cfg["http_client"] = http_cfg
            initial_concurrency = adaptive_cfg.min_concurrency
            initial_chunk = adaptive_cfg.min_chunk

        controller = AdaptiveController(
            adaptive_cfg,
            initial_concurrency=initial_concurrency,
            initial_chunk_size=initial_chunk,
        )
        domains_cfg["chunk_size"] = controller.current_chunk_size
        http_cfg["max_concurrent_requests"] = controller.current_concurrency

        for section in ("contacts", "linkedin"):
            section_cfg = enrichment_config.get(section)
            if not isinstance(section_cfg, dict):
                continue
            section_http = section_cfg.setdefault("http_client", {})
            if isinstance(section_http, dict):
                section_http["max_concurrent_requests"] = controller.current_concurrency

        ctx["adaptive_controller"] = controller

    ctx["enrichment_config"] = enrichment_config
    ctx["enrichment_flags"] = flags
    ctx["cache_config"] = enrichment_config.get("cache", {})
    ctx["circuit_breaker_config"] = enrichment_config.get("circuit_breaker", {})

    return ctx




def _prepare_process_context(ctx: dict) -> dict:
    """Return a sanitized context dictionary safe for subprocess execution."""
    allowed = {k: v for k, v in ctx.items() if k not in {'logger', 'log_lock', 'budget_tracker', 'kpi_calculator', 'metrics_file', 'user_agent_pool'}}
    return allowed


def _run_step_in_process(step_name: str, job: dict, ctx_dict: dict) -> dict:
    """Execute a pipeline step inside a subprocess."""
    local_ctx = ctx_dict.copy()
    logger = pipeline.configure_logging(local_ctx.get('verbose', False), local_ctx.get('debug', False))
    local_ctx['logger'] = logger
    local_ctx['log_lock'] = None
    ua_path = local_ctx.get('user_agent_pool_path')
    local_ctx['user_agent_pool'] = load_user_agent_pool(ua_path)
    return _run_step(step_name, job, local_ctx)


def topo_sorted(steps, logger):
    ordered = pipeline.topo_sort(steps, STEP_DEPENDENCIES)
    requested = list(dict.fromkeys(steps))
    requested_set = set(requested)
    for step in requested:
        missing = [dep for dep in STEP_DEPENDENCIES.get(step, ()) if dep not in requested_set]
        if missing:
            pipeline.log_step_event(
                logger,
                step,
                "dependency_missing",
                status="WARN",
                missing=",".join(sorted(missing)),
            )
    return ordered


def build_execution_batches(steps: Sequence[str]) -> List[List[str]]:
    """Group pipeline steps into batches that can run in parallel."""
    pending = [step for step in steps]
    completed: Set[str] = set()
    batches: List[List[str]] = []
    step_set = set(steps)
    deps: Dict[str, Set[str]] = {
        step: set(STEP_DEPENDENCIES.get(step, ())) & step_set for step in steps
    }

    while pending:
        ready: List[str] = []
        for step in pending:
            if deps[step] <= completed:
                ready.append(step)
        if not ready:
            raise ValueError("No executable steps found; dependency cycle suspected")
        batches.append(ready)
        completed.update(ready)
        pending = [step for step in pending if step not in ready]

    return batches

def explain(steps):
    print("DAG:")
    for i, step in enumerate(steps, 1):
        print(f"{i:02d}. {step}")


def _normalize_naf_codes(values):
    """Return sanitized, deduplicated NAF codes while preserving order."""
    normalized = []
    seen = set()
    for raw in values or []:
        if raw is None:
            continue
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace(";", ",").split(",")]
        else:
            parts = [str(raw).strip()]
        for part in parts:
            if not part:
                continue
            if part in seen:
                continue
            seen.add(part)
            normalized.append(part)
    return normalized


def prepare_multi_naf_runs(job, naf_codes, base_output_dir):
    """Build job variants for each NAF code with dedicated output dirs."""
    base_path = Path(base_output_dir).expanduser().resolve()
    prepared = []
    for code in _normalize_naf_codes(naf_codes):
        job_variant = copy.deepcopy(job)
        job_variant.pop("_job_path", None)
        filters = job_variant.setdefault("filters", {})
        filters["naf_include"] = [code]
        niche_name = create_job.generate_niche_name(code)
        job_variant["niche"] = niche_name
        output_cfg = job_variant.setdefault("output", {})
        outdir_path = base_path / niche_name
        output_cfg["dir"] = str(outdir_path)
        prepared.append(
            {
                "naf_code": code,
                "job": job_variant,
                "outdir": outdir_path,
                "niche": niche_name,
            }
        )
    return prepared


def run_batch_jobs(args, logger=None):
    """Process multiple NAF codes in batch mode."""
    logger = logger or pipeline.configure_logging(args.verbose, getattr(args, 'debug', False))

    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    output_base_dir = Path(args.output_dir).expanduser().resolve()
    template_path = args.template or (Path(__file__).parent / "job_template.yaml")

    io.ensure_dir(jobs_dir)
    io.ensure_dir(output_base_dir)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Generating %d job files...", len(args.naf_codes))
    job_files = create_job.generate_batch_jobs(
        args.naf_codes,
        jobs_dir,
        template_path,
        args.profile,
    )

    if args.dry_run:
        logger.info("Dry run mode: jobs generated but not executed")
        logger.info("Generated job files: %s", [str(f) for f in job_files])

        return {
            "status": "OK",
            "total_jobs": len(job_files),
            "successful_jobs": 0,
            "failed_jobs": 0,
            "elapsed": 0.0,
            "results": [],
            "dry_run": True,
        }


    batch_start = time.time()
    results = []
    failed_jobs = []

    logger.info("Executing %d jobs sequentially...", len(job_files))

    for i, job_file in enumerate(job_files, 1):
        naf_code = args.naf_codes[i - 1]
        niche_name = create_job.generate_niche_name(naf_code)
        job_output_dir = output_base_dir / niche_name

        logger.info("[%d/%d] Processing NAF %s...", i, len(job_files), naf_code)

        cmd_args = [
            sys.executable,
            "builder_cli.py",
            "run-profile",
            "--job",
            str(job_file),
            "--input",
            str(input_path),
            "--out",
            str(job_output_dir),
            "--profile",
            args.profile,
            "--enrichment-config",
            str(args.enrichment_config),
        ]

        if args.sample > 0:
            cmd_args.extend(["--sample", str(args.sample)])
        if args.workers != 8:
            cmd_args.extend(["--workers", str(args.workers)])
        if args.max_ram_mb > 0:
            cmd_args.extend(["--max-ram-mb", str(args.max_ram_mb)])
        if args.verbose:
            cmd_args.append("--verbose")
        if getattr(args, 'debug', False):
            cmd_args.append("--debug")
        if getattr(args, 'use_domains', None) is True:
            cmd_args.append("--use-domains")
        elif getattr(args, 'use_domains', None) is False:
            cmd_args.append("--no-domains")
        if getattr(args, 'use_contacts', None) is True:
            cmd_args.append("--use-contacts")
        elif getattr(args, 'use_contacts', None) is False:
            cmd_args.append("--no-contacts")
        if getattr(args, 'use_linkedin', None) is True:
            cmd_args.append("--use-linkedin")
        elif getattr(args, 'use_linkedin', None) is False:
            cmd_args.append("--no-linkedin")
        if getattr(args, 'use_correlation', None) is True:
            cmd_args.append("--use-correlation")
        elif getattr(args, 'use_correlation', None) is False:
            cmd_args.append("--no-correlation")
        if getattr(args, 'use_contacts_ai', None) is True:
            cmd_args.append("--use-contacts-ai")
        elif getattr(args, 'use_contacts_ai', None) is False:
            cmd_args.append("--no-contacts-ai")
        if getattr(args, 'use_metrics_export', None) is True:
            cmd_args.append("--use-metrics-export")
        elif getattr(args, 'use_metrics_export', None) is False:
            cmd_args.append("--no-metrics-export")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd_args,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            elapsed = time.time() - start_time

            if result.returncode == 0:
                logger.info("NAF %s completed successfully in %.1fs", naf_code, elapsed)
                results.append({
                    "naf_code": naf_code,
                    "status": "SUCCESS",
                    "duration": elapsed,
                    "output_dir": str(job_output_dir),
                })
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error("NAF %s failed: %s", naf_code, error_msg)
                failed_jobs.append(naf_code)
                results.append({
                    "naf_code": naf_code,
                    "status": "FAILED",
                    "duration": elapsed,
                    "error": error_msg,
                })
                if not args.continue_on_error:
                    raise RuntimeError(f"NAF {naf_code} failed: {error_msg}")

        except subprocess.TimeoutExpired:
            logger.error("NAF %s timed out after 1 hour", naf_code)
            failed_jobs.append(naf_code)
            results.append({
                "naf_code": naf_code,
                "status": "TIMEOUT",
                "duration": 3600,
                "error": "Job timed out",
            })
            if not args.continue_on_error:
                raise RuntimeError(f"NAF {naf_code} timeout")

        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            logger.exception("NAF %s failed", naf_code)
            failed_jobs.append(naf_code)
            results.append({
                "naf_code": naf_code,
                "status": "ERROR",
                "duration": 0,
                "error": str(exc),
            })
            if not args.continue_on_error:
                raise

    successful = [r for r in results if r["status"] == "SUCCESS"]
    total_elapsed = round(time.time() - batch_start, 1)
    logger.info("Batch processing completed: %d/%d jobs successful", len(successful), len(args.naf_codes))
    logger.info("Batch execution finished in %.1fs", total_elapsed)
    if failed_jobs:
        logger.warning("Failed NAF codes: %s", failed_jobs)

    return {
        "status": "OK" if not failed_jobs or args.continue_on_error else "FAILED",
        "total_jobs": len(args.naf_codes),
        "successful_jobs": len(successful),
        "failed_jobs": len(failed_jobs),
        "elapsed": total_elapsed,
        "results": results
    }

def execute_steps(args, job, steps, *, suppress_output=False, logger=None):
    """Run configured pipeline steps and optionally suppress console output."""
    logger = logger or pipeline.configure_logging(args.verbose, getattr(args, 'debug', False))
    ctx = build_context(args, job)
    ctx['logger'] = logger
    missing_env = config.validate_required_env(ctx.get('env', {}))
    if missing_env:
        message = (
            "Missing critical environment variables: "
            + ", ".join(missing_env)
        )
        if getattr(args, "allow_missing_secrets", False):
            logger.warning("%s; proceeding due to --allow-missing-secrets", message)
        else:
            raise config.MissingSecretError(message)
    METRICS.reset()

    steps_sorted = topo_sorted(steps, logger)
    total_steps = len(steps_sorted)

    prom_ctx = ctx.get("_prometheus")
    if isinstance(prom_ctx, dict) and prometheus_exporter:
        try:
            profile_name = getattr(args, "profile", None) or job.get("profile")
            prometheus_exporter.set_run_metadata(
                run_id=prom_ctx["run_id"],
                job_name=prom_ctx["job_name"],
                profile=profile_name,
                total_steps=total_steps,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Prometheus run metadata export failed", exc_info=True)

    step_indices = {name: index for index, name in enumerate(steps_sorted, start=1)}
    step_positions = {name: f"{index}/{total_steps}" for name, index in step_indices.items()}
    batches = build_execution_batches(steps_sorted)

    parallel_candidate = bool(ctx.get('parallel')) and ctx.get('workers', 1) > 1
    parallel_mode = ctx.get('parallel_mode', 'thread')
    use_process = parallel_candidate and parallel_mode == 'process'
    if use_process and (ctx.get('budget_tracker') or ctx.get('kpi_calculator')):
        logger.warning("Process-based parallelism not supported with budget tracking or KPI calculation; falling back to threads")
        use_process = False
    use_thread = parallel_candidate and (parallel_mode == 'thread' or (parallel_mode == 'process' and not use_process))
    parallel_enabled = use_thread or use_process

    if ctx.get('debug'):
        logger.info("[DEBUG] Pipeline configuration:")
        logger.info(f"[DEBUG] - Profile: {getattr(args, 'profile', 'N/A')}")
        logger.info(f"[DEBUG] - Total steps: {total_steps}")
        logger.info(f"[DEBUG] - Steps: {', '.join(steps_sorted)}")
        logger.info(f"[DEBUG] - Output directory: {ctx['outdir']}")
        logger.info(f"[DEBUG] - Run ID: {ctx['run_id']}")
        if ctx.get('verbose'):
            logger.debug(f"[VERBOSE] Job configuration: {json.dumps(job, indent=2, ensure_ascii=False)}")
        if parallel_enabled:
            logger.info("[DEBUG] Parallel execution enabled with up to %s workers", ctx.get('workers', 1))

    results: List[dict] = []
    ctx['metrics'] = {
        'parallel': parallel_enabled,
        'worker_count': ctx.get('workers', 1),
    }

    pipeline.log_step_event(
        logger,
        'pipeline',
        'start',
        status='OK',
        total_steps=total_steps,
    )
    start_time = time.time()

    for batch in batches:
        enabled_batch: List[str] = []
        for name in batch:
            if ctx.get('debug'):
                logger.info("[DEBUG] Queuing step %s: %s", step_positions[name], name)
            pipeline.log_step_event(
                logger,
                name,
                'queued',
                status='OK',
                position=step_positions[name],
            )
            if _is_enrichment_step_enabled(name, ctx):
                enabled_batch.append(name)
            else:
                status = _mark_enrichment_step_disabled(name, ctx)
                results.append(status)

        if not enabled_batch:
            continue

        if use_process and len(enabled_batch) > 1:
            max_workers = max(1, min(ctx.get('workers', 1), len(enabled_batch)))
            results_map: Dict[str, dict] = {}
            payload_ctx = _prepare_process_context(ctx)
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(_run_step_in_process, name, copy.deepcopy(job), copy.deepcopy(payload_ctx)): name
                        for name in enabled_batch
                    }
                    for future in as_completed(future_map):
                        name = future_map[future]
                        result = future.result()
                        _register_resume_outputs(ctx, result)
                        results_map[name] = result
                        if ctx.get('debug'):
                            logger.info("[DEBUG] Completed step %s: %s (%s)", step_positions[name], name, result['status'])
            except Exception:
                for future in future_map:
                    future.cancel()
                raise

            batch_results = [results_map[name] for name in enabled_batch]
            _apply_adaptive_feedback(ctx, batch_results)
            for name in enabled_batch:
                results.append(results_map[name])
        elif use_thread and len(enabled_batch) > 1:
            max_workers = max(1, min(ctx.get('workers', 1), len(enabled_batch)))
            results_map: Dict[str, dict] = {}
            future_map: Dict = {}
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(_run_step, name, job, ctx): name for name in enabled_batch}
                    for future in as_completed(future_map):
                        name = future_map[future]
                        result = future.result()
                        _register_resume_outputs(ctx, result)
                        results_map[name] = result
                        if ctx.get('debug'):
                            logger.info("[DEBUG] Completed step %s: %s (%s)", step_positions[name], name, result['status'])
            except Exception:
                for future in future_map:
                    future.cancel()
                raise

            batch_results = [results_map[name] for name in enabled_batch]
            _apply_adaptive_feedback(ctx, batch_results)
            for name in enabled_batch:
                results.append(results_map[name])
        else:
            batch_results: List[dict] = []
            for name in enabled_batch:
                result = _run_step(name, job, ctx)
                _register_resume_outputs(ctx, result)
                results.append(result)
                batch_results.append(result)
                if ctx.get('debug'):
                    logger.info("[DEBUG] Completed step %s: %s (%s)", step_positions[name], name, result['status'])
            _apply_adaptive_feedback(ctx, batch_results)

    elapsed = round(time.time() - start_time, 1)
    ctx['metrics']['elapsed'] = elapsed

    if ctx.get('debug'):
        logger.info(f"[DEBUG] Pipeline completed in {elapsed}s")
        successful_steps = sum(1 for r in results if r['status'] in ('OK', 'SKIPPED', 'WARN'))
        logger.info(f"[DEBUG] Step summary: {successful_steps}/{len(results)} successful")

    pipeline.log_step_event(
        logger,
        'pipeline',
        'end',
        status='OK',
        duration=elapsed,
    )

    kpi_calculator = ctx.get('kpi_calculator')
    budget_tracker = ctx.get('budget_tracker')
    final_kpis = None

    if kpi_calculator:
        if ctx.get('debug'):
            logger.info('[DEBUG] Calculating final KPIs...')
        try:
            final_kpis = kpi_calculator.calculate_final_kpis(ctx, results)
            kpi_status = 'OK' if final_kpis['all_kpis_met'] else 'WARN'

            if ctx.get('debug'):
                logger.info(f"[DEBUG] KPI calculation result: {kpi_status}")
                logger.info(f"[DEBUG] KPIs met: {final_kpis['all_kpis_met']}")
                if ctx.get('verbose'):
                    logger.debug(f"[VERBOSE] KPI details: {json.dumps(final_kpis, indent=2, ensure_ascii=False)}")

            pipeline.log_step_event(
                logger,
                'kpi_calculation',
                'end',
                status=kpi_status,
                **{f"kpi_{k}": v for k, v in final_kpis['actual_kpis'].items()},
            )

            kpi_status_obj = {
                'step': 'kpi_calculation',
                'status': kpi_status,
                'out': final_kpis,
                'duration_s': 0,
            }
            log_lock = ctx.get("log_lock")
            with (log_lock if log_lock else nullcontext()):
                io.log_json(ctx["logs"], kpi_status_obj)

        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            logger.exception("KPI calculation failed")
            pipeline.log_step_event(logger, 'kpi_calculation', 'error', status='FAIL', error=str(exc))
            raise RuntimeError("KPI calculation failed") from exc

    if budget_tracker:
        final_budget_stats = budget_tracker.get_current_stats()
        if ctx.get('debug'):
            logger.info(f"[DEBUG] Final budget statistics: {final_budget_stats}")

        budget_status = {
            'step': 'budget_summary',
            'status': 'OK',
            'out': final_budget_stats,
            'duration_s': 0,
        }
        log_lock = ctx.get("log_lock")
        with (log_lock if log_lock else nullcontext()):
            io.log_json(ctx["logs"], budget_status)

    output_data = None
    kpi_msg = ''
    if final_kpis and not final_kpis['all_kpis_met']:
        kpi_msg = ' (KPIs NOT MET)'
    elif final_kpis and final_kpis['all_kpis_met']:
        kpi_msg = ' (KPIs MET)'

    budget_msg = ''
    if budget_tracker:
        stats = budget_tracker.get_current_stats()
        budget_msg = f" (req: {stats['http_requests']}/{stats['max_http_requests']}, bytes: {stats['http_bytes']}/{stats['max_http_bytes']})"

    success_msg = f"RUN {ctx['run_id']} DONE -> {ctx['outdir']} (elapsed={elapsed}s){kpi_msg}{budget_msg}"

    if args.json:
        output_data = {
            'run_id': ctx['run_id'],
            'results': results,
            'outdir': ctx['outdir'],
        }
        if final_kpis:
            output_data['kpis'] = final_kpis
        if budget_tracker:
            output_data['budget_stats'] = budget_tracker.get_current_stats()
        if not suppress_output:
            print(json.dumps(output_data, ensure_ascii=False))
    else:
        if not suppress_output:
            print(success_msg)

    metrics_target = Path(ctx.get('metrics_file') or "reports/report_metrics.json")
    export_path = None
    try:
        export_path = METRICS.export_json(metrics_target)
    except OSError as exc:  # pragma: no cover - filesystem issues
        logger.warning("Unable to write metrics report %s: %s", metrics_target, exc)

    metrics_summary = METRICS.summary()
    ctx['metrics']['summary'] = metrics_summary
    if export_path is not None:
        ctx['metrics']['report_file'] = str(export_path)
        if ctx.get('debug'):
            logger.info("[DEBUG] Metrics exported to %s", export_path)

    summary_line = METRICS.format_summary()
    if not suppress_output:
        stream = sys.stdout if not getattr(args, 'json', False) else sys.stderr
        print(summary_line, file=stream)
    else:
        logger.info(summary_line)

    if ctx.get('debug'):
        failed_steps = [r for r in results if r['status'] not in ('OK', 'SKIPPED', 'WARN')]
        if failed_steps:
            logger.info(f"[DEBUG] Failed steps: {[s['step'] for s in failed_steps]}")
        else:
            logger.info('[DEBUG] All steps completed successfully')

    if ctx.get('verbose'):
        logger.debug('[VERBOSE] Complete run summary:')
        for r in results:
            logger.debug(f"[VERBOSE] - {r['step']}: {r['status']} ({r['duration_s']}s)")
        if final_kpis:
            logger.debug(f"[VERBOSE] Final KPIs: {final_kpis['actual_kpis']}")
        if budget_tracker:
            logger.debug(f"[VERBOSE] Final budget usage: {budget_tracker.get_current_stats()}")
    return {
        'status': 'OK',
        'results': results,
        'final_kpis': final_kpis,
        'ctx': ctx,
        'elapsed': elapsed,
        'output_data': output_data,
        'success_message': success_msg,
    }
def run_profile_multi_nafs(args, job, profile, steps, logger):
    '''Execute the configured profile for each requested NAF code.'''
    base_outdir = Path(args.out).expanduser().resolve()
    io.ensure_dir(base_outdir)
    runs = prepare_multi_naf_runs(job, getattr(args, 'naf_codes', []), base_outdir)
    if not runs:
        raise ValueError('No NAF codes provided for multi-NAF execution')

    results = []
    failures = []
    suppress_output = bool(args.json)
    total = len(runs)

    for index, run in enumerate(runs, 1):
        run_job = copy.deepcopy(run['job'])
        if profile:
            run_job['profile'] = profile
        logger.info(f"[MULTI-NAF] Starting {run['naf_code']} ({index}/{total})")

        run_args = copy.deepcopy(args)
        run_args.naf_codes = None
        run_args.out = str(run['outdir'])
        run_args.json = False if suppress_output else args.json
        run_args.run_id = None

        try:
            result = execute_steps(run_args, run_job, steps, suppress_output=suppress_output, logger=logger)
            results.append(
                {
                    'naf_code': run['naf_code'],
                    'status': 'OK',
                    'run_id': result['ctx']['run_id'],
                    'outdir': result['ctx']['outdir'],
                    'elapsed': result['elapsed'],
                    'kpis': result['final_kpis'],
                }
            )
        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            logger.exception('[MULTI-NAF] %s failed', run['naf_code'])
            failures.append({'naf_code': run['naf_code'], 'error': str(exc)})
            if not getattr(args, 'continue_on_error', False):
                raise

    status = 'OK'
    if failures:
        status = 'FAILED' if len(failures) == total else 'PARTIAL'
        logger.warning(f"Failed NAF codes: {[f['naf_code'] for f in failures]}")
    summary = {
        'status': status,
        'profile': profile,
        'total_runs': total,
        'successful_runs': len(results),
        'failed_runs': failures,
        'results': results,
    }
    summary_line = f"MULTI-NAF {len(results)}/{total} runs succeeded ({status})"
    if args.json:
        summary['message'] = summary_line
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(summary_line)
        if failures:
            print(f"Failed NAF codes: {[f['naf_code'] for f in failures]}", file=sys.stderr)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='builder_cli')
    sub = parser.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--job', required=True)
    common.add_argument('--out', required=True)
    common.add_argument('--input')
    common.add_argument('--run-id')
    common.add_argument(
        '--enrichment-config',
        type=Path,
        default=Path('config/enrichment.yaml'),
        help='Path to enrichment config YAML file (default: config/enrichment.yaml)',
    )
    common.add_argument('--dry-run', action='store_true')
    common.add_argument('--sample', type=int, default=0)
    common.add_argument('--concurrency', type=int, default=0,
                        help='Maximum concurrent domains for crawling (overrides job config)')
    common.add_argument('--per-domain-rps', type=float, default=0.0,
                        help='Override crawl rate limit (requests per second per domain)')
    common.add_argument(
        '--time-budget-min',
        type=int,
        default=TIME_BUDGET_DEFAULT,
        help=f'Maximum run duration in minutes (default: {TIME_BUDGET_DEFAULT})',
    )
    common.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Maximum parallel workers for step execution (default: 8)',
    )
    common.add_argument('--json', action='store_true')
    common.add_argument('--resume', action='store_true')
    common.add_argument('--verbose', action='store_true', help='Enable verbose logging with all process details')
    common.add_argument('--debug', action='store_true', help='Enable debug mode with important debug information')
    common.add_argument(
        '--allow-missing-secrets',
        action='store_true',
        help='Continue even if critical environment variables are missing (may disable some enrichments)',
    )
    common.add_argument(
        '--max-ram-mb',
        type=int,
        default=RAM_BUDGET_DEFAULT,
        help=f'Maximum RAM budget in MB (default: {RAM_BUDGET_DEFAULT} - 0 disables)',
    )
    common.add_argument('--metrics-file', type=Path, help='Write pipeline metrics JSON to this file')
    common.add_argument('--prometheus-port', type=int, default=0,
                        help='Expose Prometheus metrics on this port (0 disables export)')
    common.add_argument('--prometheus-address', default='0.0.0.0',
                        help='Bind address for the Prometheus metrics server')
    common.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution for independent steps')
    common.add_argument('--parallel-mode', choices=['thread', 'process'], default='thread',
                        help='Parallel execution mode (default: thread)')
    common.add_argument('--serp-timeout-sec', type=float, default=8.0,
                         help='Timeout (seconds) for SERP requests (default: 8)')
    common.add_argument('--max-pages-per-domain', type=int, default=12,
                         help='Maximum pages to crawl per domain (default: 12)')
    common.add_argument('--crawl-time-budget-min', type=float, default=60.0,
                         help='Crawl time budget in minutes (default: 60)')
    common.add_argument('--user-agent-pool', type=Path,
                         help='Optional path to a file containing User-Agent strings (one per line)')
    common.add_argument('--respect-robots', dest='respect_robots', action='store_true',
                         help='Respect robots.txt directives (default)')
    common.add_argument('--no-respect-robots', dest='respect_robots', action='store_false',
                         help='Ignore robots.txt directives for crawling')
    common.add_argument('--use-domains', dest='use_domains', action='store_true',
                        help='Force-enable domain enrichment (overrides config)')
    common.add_argument('--no-domains', dest='use_domains', action='store_false',
                        help='Disable domain enrichment regardless of config')
    common.add_argument('--use-contacts', dest='use_contacts', action='store_true',
                        help='Force-enable contact enrichment (overrides config)')
    common.add_argument('--no-contacts', dest='use_contacts', action='store_false',
                        help='Disable contact enrichment regardless of config')
    common.add_argument('--use-linkedin', dest='use_linkedin', action='store_true',
                        help='Force-enable LinkedIn enrichment (overrides config)')
    common.add_argument('--no-linkedin', dest='use_linkedin', action='store_false',
                        help='Disable LinkedIn enrichment regardless of config')
    common.add_argument('--use-correlation', dest='use_correlation', action='store_true',
                        help='Force-enable correlation scoring (overrides config)')
    common.add_argument('--no-correlation', dest='use_correlation', action='store_false',
                        help='Disable correlation scoring regardless of config')
    common.add_argument('--use-contacts-ai', dest='use_contacts_ai', action='store_true',
                        help='Force-enable AI-assisted contact enrichment (overrides config)')
    common.add_argument('--no-contacts-ai', dest='use_contacts_ai', action='store_false',
                        help='Disable AI-assisted contact enrichment')
    common.add_argument('--use-metrics-export', dest='use_metrics_export', action='store_true',
                        help='Export metrics as a pipeline step (overrides config)')
    common.add_argument('--no-metrics-export', dest='use_metrics_export', action='store_false',
                        help='Skip metrics export regardless of config')
    common.set_defaults(
        respect_robots=None,
        use_domains=None,
        use_contacts=None,
        use_linkedin=None,
        use_correlation=None,
        use_contacts_ai=None,
        use_metrics_export=None,
    )


    run_step_parser = sub.add_parser('run-step', parents=[common])
    run_step_parser.add_argument('--step', required=True)

    run_profile_parser = sub.add_parser('run-profile', parents=[common])
    run_profile_parser.add_argument('--profile', choices=list(PROFILES.keys()), required=True)
    run_profile_parser.add_argument('--explain', action='store_true')
    run_profile_parser.add_argument('--naf', dest='naf_codes', action='append',
                                    help='NAF code(s) to process (repeat option or use comma-separated values)')
    run_profile_parser.add_argument('--continue-on-error', action='store_true',
                                    help='Continue processing other NAF codes if one fails')

    batch_parser = sub.add_parser('batch', help='Generate and run jobs for multiple NAF codes')
    batch_parser.add_argument('--naf', dest='naf_codes', action='append', required=True,
                             help='NAF code(s) to process (can be used multiple times)')
    batch_parser.add_argument('--template', type=Path,
                             help='Path to job template file (default: job_template.yaml)')
    batch_parser.add_argument('--profile', choices=list(PROFILES.keys()), default='quick',
                             help='Profile to use for jobs (default: quick)')
    batch_parser.add_argument('--input', required=True, help='Input file for processing')
    batch_parser.add_argument('--output-dir', '--out-dir', required=True,
                             help='Base output directory for all jobs')
    batch_parser.add_argument('--jobs-dir', default='jobs_generated',
                             help='Directory to store generated job files (default: jobs_generated)')
    batch_parser.add_argument('--dry-run', action='store_true', help="Generate jobs but don't run them")
    batch_parser.add_argument('--sample', type=int, default=0, help='Sample size for testing')
    batch_parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    batch_parser.add_argument(
        '--enrichment-config',
        type=Path,
        default=Path('config/enrichment.yaml'),
        help='Path to enrichment config YAML file (default: config/enrichment.yaml)',
    )
    batch_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging with all process details')
    batch_parser.add_argument('--debug', action='store_true', help='Enable debug mode with important debug information')
    batch_parser.add_argument(
        '--max-ram-mb',
        type=int,
        default=RAM_BUDGET_DEFAULT,
        help=f'Maximum RAM budget in MB (default: {RAM_BUDGET_DEFAULT} - 0 disables)',
    )
    batch_parser.add_argument('--continue-on-error', action='store_true', help='Continue processing other NAF codes if one fails')
    batch_parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    batch_parser.add_argument('--use-domains', dest='use_domains', action='store_true',
                              help='Force-enable domain enrichment for generated jobs')
    batch_parser.add_argument('--no-domains', dest='use_domains', action='store_false',
                              help='Disable domain enrichment for generated jobs')
    batch_parser.add_argument('--use-contacts', dest='use_contacts', action='store_true',
                              help='Force-enable contact enrichment for generated jobs')
    batch_parser.add_argument('--no-contacts', dest='use_contacts', action='store_false',
                              help='Disable contact enrichment for generated jobs')
    batch_parser.add_argument('--use-linkedin', dest='use_linkedin', action='store_true',
                              help='Force-enable LinkedIn enrichment for generated jobs')
    batch_parser.add_argument('--no-linkedin', dest='use_linkedin', action='store_false',
                              help='Disable LinkedIn enrichment for generated jobs')
    batch_parser.add_argument('--use-correlation', dest='use_correlation', action='store_true',
                              help='Force-enable correlation scoring for generated jobs')
    batch_parser.add_argument('--no-correlation', dest='use_correlation', action='store_false',
                              help='Disable correlation scoring for generated jobs')
    batch_parser.add_argument('--use-contacts-ai', dest='use_contacts_ai', action='store_true',
                              help='Force-enable AI-assisted contact enrichment for generated jobs')
    batch_parser.add_argument('--no-contacts-ai', dest='use_contacts_ai', action='store_false',
                              help='Disable AI-assisted contact enrichment for generated jobs')
    batch_parser.add_argument('--use-metrics-export', dest='use_metrics_export', action='store_true',
                              help='Export metrics as part of generated jobs')
    batch_parser.add_argument('--no-metrics-export', dest='use_metrics_export', action='store_false',
                              help='Disable metrics export for generated jobs')
    batch_parser.set_defaults(
        use_domains=None,
        use_contacts=None,
        use_linkedin=None,
        use_correlation=None,
        use_contacts_ai=None,
        use_metrics_export=None,
    )

    sub.add_parser('resume', parents=[common])

    args = parser.parse_args(argv)
    logger = pipeline.configure_logging(args.verbose, getattr(args, 'debug', False))

    if hasattr(args, "enrichment_config"):
        args.enrichment_config = Path(args.enrichment_config).expanduser()
        args.enrich_cfg = load_enrichment_config(args.enrichment_config)

    try:
        if args.cmd == 'batch':
            result = run_batch_jobs(args, logger=logger)
            if getattr(args, 'json', False):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        job = load_job(args.job)

        if args.cmd == 'run-step':
            steps = [args.step]
            profile = None
        elif args.cmd in ('run-profile', 'resume'):
            profile = getattr(args, 'profile', None) or job.get('profile') or 'quick'
            steps = PROFILES[profile]
        else:
            profile = None
            steps = []

        if args.cmd == 'run-profile' and getattr(args, 'explain', False):
            explain(steps)
            return 0

        if args.cmd == 'run-profile' and getattr(args, 'naf_codes', None):
            run_profile_multi_nafs(args, job, profile, steps, logger)
            return 0

        execute_steps(args, job, steps, logger=logger)
        return 0

    except KeyboardInterrupt:
        logger.warning('Execution interrupted')
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, io.IoError) as exc:
        logger.error(str(exc))
        return 1
    except Exception:
        logger.exception('Unhandled error')
        return 1



if __name__ == "__main__":
    sys.exit(main())
