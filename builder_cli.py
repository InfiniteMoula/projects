import argparse
import json
import os
import time
import traceback
import uuid
from pathlib import Path

import psutil
import yaml
from jsonschema import validate as js_validate

from utils import config, io, pipeline, budget_middleware

STEP_REGISTRY = {
    "dumps.collect": "dumps.collect_dump:run",
    "api.collect": "api.collect_api:run",
    "http.static": "nethttp.collect_http_static:run",
    "http.sitemap": "nethttp.collect_sitemap:run",
    "headless.collect": "headless.collect_headless:run",
    "feeds.collect": "feeds.collect_rss:run",
    "pdf.collect": "pdf.collect_pdf:run",
    "parse.html": "parse.parse_html:run",
    "parse.jsonld": "parse.parse_jsonld:run",
    "parse.pdf": "parse.parse_pdf:run",
    "normalize.standardize": "normalize.standardize:run",
    "enrich.domain": "enrich.domain_discovery:run",
    "enrich.site": "enrich.site_probe:run",
    "enrich.dns": "enrich.dns_checks:run",
    "enrich.email": "enrich.email_heuristics:run",
    "enrich.phone": "enrich.phone_checks:run",
    "quality.checks": "quality.checks:run",
    "quality.dedupe": "quality.dedupe:run",
    "quality.score": "quality.score:run",
    "package.export": "package.exporter:run",
}

STEP_DEPENDENCIES = {
    "http.static": {"dumps.collect"},
    "http.sitemap": {"dumps.collect"},
    "headless.collect": {"http.static"},
    "pdf.collect": {"dumps.collect"},
    "parse.html": {"http.static", "headless.collect"},
    "parse.jsonld": {"http.static", "http.sitemap"},
    "parse.pdf": {"pdf.collect"},
    "normalize.standardize": {
        "dumps.collect",
        "api.collect",
        "parse.html",
        "parse.jsonld",
        "parse.pdf",
    },
    "enrich.domain": {"normalize.standardize"},
    "enrich.site": {"enrich.domain"},
    "enrich.dns": {"enrich.domain"},
    "enrich.email": {"enrich.dns"},
    "enrich.phone": {"enrich.email"},
    "quality.checks": {"normalize.standardize"},
    "quality.dedupe": {"enrich.email", "normalize.standardize"},
    "quality.score": {"quality.dedupe"},
    "package.export": {"quality.score"},
}

PROFILES = {
    "quick": [
        "dumps.collect",
        "api.collect",
        "normalize.standardize",
        "quality.checks",
        "quality.dedupe",
        "quality.score",
        "package.export",
    ],
    "standard": [
        "dumps.collect",
        "api.collect",
        "http.static",
        "http.sitemap",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "quality.checks",
        "quality.dedupe",
        "quality.score",
        "package.export",
    ],
    "deep": [
        "dumps.collect",
        "api.collect",
        "http.static",
        "http.sitemap",
        "headless.collect",
        "pdf.collect",
        "parse.pdf",
        "parse.html",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "quality.checks",
        "quality.dedupe",
        "quality.score",
        "package.export",
    ],
}


def load_job(job_path: os.PathLike[str] | str):
    job_file = Path(job_path).expanduser().resolve()
    with job_file.open("r", encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    
    # Check for schema validation if schema file exists
    schema_path = job_file.with_name("job_schema.json")
    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        js_validate(job, schema)
    
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
    
    # Check time budget before starting step
    if budget_tracker:
        try:
            budget_tracker.check_time_budget()
        except budget_middleware.BudgetExceededError as exc:
            status = {
                "step": step_name,
                "status": "BUDGET_EXCEEDED",
                "error": str(exc),
                "duration_s": 0,
            }
            pipeline.log_step_event(logger, step_name, "budget_exceeded", status="BUDGET_EXCEEDED", error=str(exc))
            io.log_json(context["logs"], status)
            raise RuntimeError(f"step {step_name} budget exceeded: {exc}")
    
    fn = pipeline.resolve_step(step_name, STEP_REGISTRY)
    pipeline.log_step_event(logger, step_name, "start")
    started = time.time()
    try:
        _check_ram(context)
        
        # Add budget tracker to context for steps to use
        step_context = context.copy()
        if budget_tracker:
            step_context["budget_tracker"] = budget_tracker
            step_context["request_tracker"] = lambda size: budget_tracker.track_http_request(size)
        
        out = fn(cfg, step_context) or {}
        _check_ram(context)
        
        # Check budgets after step completion
        if budget_tracker:
            budget_tracker.check_time_budget()
            
        status = {
            "step": step_name,
            "status": out.get("status") or "OK",
            "out": out,
            "duration_s": round(time.time() - started, 3),
        }
        
        # Add budget stats to status
        if budget_tracker:
            status["budget_stats"] = budget_tracker.get_current_stats()
            
    except budget_middleware.BudgetExceededError as exc:
        status = {
            "step": step_name,
            "status": "BUDGET_EXCEEDED", 
            "error": str(exc),
            "duration_s": round(time.time() - started, 3),
        }
        if budget_tracker:
            status["budget_stats"] = budget_tracker.get_current_stats()
    except Exception as exc:
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
    io.log_json(context["logs"], status)
    if context.get("verbose"):
        logger.debug("step_result %s", json.dumps(status, ensure_ascii=False))
    if status["status"] not in ("OK", "SKIPPED", "WARN"):
        raise RuntimeError(f"step {step_name} failed")
    return status


def build_context(args, job):
    run_id = args.run_id or uuid.uuid4().hex[:12]
    outdir_path = Path(args.out).expanduser().resolve()
    io.ensure_dir(outdir_path)
    logs_dir = io.ensure_dir(outdir_path / "logs")

    input_path = None
    if args.input:
        input_candidate = Path(args.input).expanduser().resolve()
        if not input_candidate.exists():
            raise FileNotFoundError(f"Input path not found: {input_candidate}")
        input_path = input_candidate

    ctx = {
        "run_id": run_id,
        "outdir": str(outdir_path),
        "outdir_path": outdir_path,
        "logs": str((logs_dir / f"{run_id}.jsonl")),
        "logs_path": logs_dir / f"{run_id}.jsonl",
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
        "max_ram_mb": args.max_ram_mb,
    }
    
    # Initialize budget tracker if budgets are configured
    budget_tracker = budget_middleware.create_budget_tracker(job)
    if budget_tracker:
        ctx["budget_tracker"] = budget_tracker
        
    # Initialize KPI calculator if targets are configured  
    kpi_calculator = budget_middleware.create_kpi_calculator(job)
    if kpi_calculator:
        ctx["kpi_calculator"] = kpi_calculator
    
    return ctx


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


def explain(steps):
    print("DAG:")
    for i, step in enumerate(steps, 1):
        print(f"{i:02d}. {step}")


def main():
    parser = argparse.ArgumentParser(prog="builder_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--job", required=True)
    common.add_argument("--out", required=True)
    common.add_argument("--input")
    common.add_argument("--run-id")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--sample", type=int, default=0)
    common.add_argument("--time-budget-min", type=int, default=0)
    common.add_argument("--workers", type=int, default=8)
    common.add_argument("--json", action="store_true")
    common.add_argument("--resume", action="store_true")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--max-ram-mb", type=int, default=0)  # 0 = desactive

    run_step_parser = sub.add_parser("run-step", parents=[common])
    run_step_parser.add_argument("--step", required=True)

    run_profile_parser = sub.add_parser("run-profile", parents=[common])
    run_profile_parser.add_argument("--profile", choices=list(PROFILES.keys()), required=True)
    run_profile_parser.add_argument("--explain", action="store_true")

    sub.add_parser("resume", parents=[common])

    args = parser.parse_args()
    job = load_job(args.job)

    if args.cmd == "run-step":
        steps = [args.step]
    elif args.cmd in ("run-profile", "resume"):
        profile = getattr(args, "profile", None) or job.get("profile") or "quick"
        steps = PROFILES[profile]
    else:
        steps = []

    logger = pipeline.configure_logging(args.verbose)

    if args.cmd == "run-profile" and args.explain:
        explain(steps)
        return

    ctx = build_context(args, job)
    ctx["logger"] = logger

    steps_sorted = topo_sorted(steps, logger)
    results = []

    pipeline.log_step_event(
        logger,
        "pipeline",
        "start",
        status="OK",
        total_steps=len(steps_sorted),
    )
    start_time = time.time()
    for index, name in enumerate(steps_sorted, start=1):
        pipeline.log_step_event(
            logger,
            name,
            "queued",
            status="OK",
            position=f"{index}/{len(steps_sorted)}",
        )
        result = _run_step(name, job, ctx)
        results.append(result)

    elapsed = round(time.time() - start_time, 1)
    pipeline.log_step_event(
        logger,
        "pipeline",
        "end",
        status="OK",
        duration=elapsed,
    )

    # Calculate final KPIs if configured
    kpi_calculator = ctx.get("kpi_calculator")
    budget_tracker = ctx.get("budget_tracker")
    final_kpis = None
    
    if kpi_calculator:
        try:
            final_kpis = kpi_calculator.calculate_final_kpis(ctx, results)
            pipeline.log_step_event(
                logger,
                "kpi_calculation",
                "end",
                status="OK" if final_kpis["all_kpis_met"] else "WARN",
                **{f"kpi_{k}": v for k, v in final_kpis["actual_kpis"].items()},
            )
            
            # Log KPI summary
            kpi_status = {
                "step": "kpi_calculation",
                "status": "OK" if final_kpis["all_kpis_met"] else "KPI_FAILED",
                "out": final_kpis,
                "duration_s": 0,
            }
            io.log_json(ctx["logs"], kpi_status)
            
        except Exception as exc:
            pipeline.log_step_event(logger, "kpi_calculation", "error", status="FAIL", error=str(exc))
    
    # Log final budget stats
    if budget_tracker:
        final_budget_stats = budget_tracker.get_current_stats()
        budget_status = {
            "step": "budget_summary",
            "status": "OK",
            "out": final_budget_stats,
            "duration_s": 0,
        }
        io.log_json(ctx["logs"], budget_status)

    if args.json:
        output_data = {
            "run_id": ctx["run_id"],
            "results": results,
            "outdir": ctx["outdir"],
        }
        
        # Add KPI results if available
        if final_kpis:
            output_data["kpis"] = final_kpis
            
        # Add budget stats if available
        if budget_tracker:
            output_data["budget_stats"] = budget_tracker.get_current_stats()
            
        print(json.dumps(output_data, ensure_ascii=False))
    else:
        kpi_msg = ""
        if final_kpis and not final_kpis["all_kpis_met"]:
            kpi_msg = " (KPIs NOT MET)"
        elif final_kpis and final_kpis["all_kpis_met"]:
            kpi_msg = " (KPIs MET)"
            
        budget_msg = ""
        if budget_tracker:
            stats = budget_tracker.get_current_stats()
            budget_msg = f" (req: {stats['http_requests']}/{stats['max_http_requests']}, bytes: {stats['http_bytes']}/{stats['max_http_bytes']})"
            
        print(f"RUN {ctx['run_id']} DONE -> {ctx['outdir']} (elapsed={elapsed}s){kpi_msg}{budget_msg}")


if __name__ == "__main__":
    main()
