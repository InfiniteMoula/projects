import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from typing import Optional, Sequence
from pathlib import Path

import psutil
import yaml
from jsonschema import ValidationError as SchemaValidationError, validate as js_validate

import create_job
from utils import budget_middleware, config, io, pipeline

STEP_REGISTRY = {
    "dumps.collect": "dumps.collect_dump:run",
    "api.collect": "api.collect_api:run",
    "api.apify": "api.apify_agents:run",
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
    "enrich.address": "enrich.address_search:run",
    "enrich.google_maps": "enrich.google_maps_search:run",
    "quality.checks": "quality.checks:run",
    "quality.dedupe": "quality.dedupe:run",
    "quality.score": "quality.score:run",
    "package.export": "package.exporter:run",
}

STEP_DEPENDENCIES = {
    "headless.collect": {"dumps.collect"},
    "feeds.collect": {"dumps.collect"},
    "pdf.collect": {"dumps.collect"},
    "parse.html": {"headless.collect"},
    "parse.jsonld": {"feeds.collect"},
    "parse.pdf": {"pdf.collect"},
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
    "enrich.google_maps": {"enrich.address"},
    "api.apify": {"enrich.address"},  # Apify agents depend on address extraction (step 7)
    "quality.checks": {"normalize.standardize"},
    "quality.dedupe": {"enrich.email", "normalize.standardize"},
    "quality.score": {"normalize.standardize"},
    "package.export": {"quality.score"},
}

PROFILES = {
    "quick": [
        "dumps.collect",
        "api.collect",
        "normalize.standardize",
        "quality.checks",
        "quality.score",
        "package.export",
    ],
    "standard": [
        "dumps.collect",
        "api.collect",
        "feeds.collect",
        "parse.jsonld",
        "normalize.standardize",
        "enrich.address",
        "api.apify",  # Apify agents after address extraction
        "enrich.google_maps",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "quality.checks",
        "quality.score",
        "package.export",
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
        "api.apify",  # Apify agents after address extraction
        "enrich.google_maps",
        "enrich.domain",
        "enrich.site",
        "enrich.dns",
        "enrich.email",
        "enrich.phone",
        "quality.checks",
        "quality.score",
        "package.export",
    ],
}


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
    io.log_json(context["logs"], status)
    if verbose:
        logger.debug(f"[VERBOSE] Complete step result for {step_name}: {json.dumps(status, ensure_ascii=False)}")
    elif debug and status.get("error"):
        logger.info(f"[DEBUG] Step '{step_name}' result: {status['status']} in {status['duration_s']}s")
    
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
        "debug": getattr(args, 'debug', False),
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
        return {"status": "OK", "generated_jobs": len(job_files), "dry_run": True}

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
                results.append(
                    {
                        "naf_code": naf_code,
                        "status": "SUCCESS",
                        "duration": elapsed,
                        "output_dir": str(job_output_dir),
                    }
                )
                continue

            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error("NAF %s failed: %s", naf_code, error_msg)
            failed_jobs.append(naf_code)
            results.append(
                {
                    "naf_code": naf_code,
                    "status": "FAILED",
                    "duration": elapsed,
                    "error": error_msg,
                }
            )
            if not args.continue_on_error:
                raise RuntimeError(f"NAF {naf_code} failed: {error_msg}")

        except subprocess.TimeoutExpired:
            logger.error("NAF %s timed out after 1 hour", naf_code)
            failed_jobs.append(naf_code)
            results.append(
                {
                    "naf_code": naf_code,
                    "status": "TIMEOUT",
                    "duration": 3600,
                    "error": "Job timed out",
                }
            )
            if not args.continue_on_error:
                raise RuntimeError(f"NAF {naf_code} timeout")

        except Exception as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            logger.exception("NAF %s failed", naf_code)
            failed_jobs.append(naf_code)
            results.append(
                {
                    "naf_code": naf_code,
                    "status": "ERROR",
                    "duration": 0,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise

    successful = [r for r in results if r["status"] == "SUCCESS"]
    logger.info(
        "Batch processing completed: %d/%d jobs successful",
        len(successful),
        len(args.naf_codes),
    )
    if failed_jobs:
        logger.warning("Failed NAF codes: %s", failed_jobs)

    return {
        "status": "OK" if not failed_jobs or args.continue_on_error else "FAILED",
        "total_jobs": len(args.naf_codes),
        "successful_jobs": len(successful),
        "failed_jobs": len(failed_jobs),
        "results": results,
    }


def execute_steps(args, job, steps, *, suppress_output=False, logger=None):
    '''Run configured pipeline steps and optionally suppress console output.'''
    logger = logger or pipeline.configure_logging(args.verbose, getattr(args, 'debug', False))
    ctx = build_context(args, job)
    ctx['logger'] = logger

    steps_sorted = topo_sorted(steps, logger)

    if ctx.get('debug'):
        logger.info(f"[DEBUG] Pipeline configuration:")
        logger.info(f"[DEBUG] - Profile: {getattr(args, 'profile', 'N/A')}")
        logger.info(f"[DEBUG] - Total steps: {len(steps_sorted)}")
        logger.info(f"[DEBUG] - Steps: {', '.join(steps_sorted)}")
        logger.info(f"[DEBUG] - Output directory: {ctx['outdir']}")
        logger.info(f"[DEBUG] - Run ID: {ctx['run_id']}")
        if ctx.get('verbose'):
            logger.debug(f"[VERBOSE] Job configuration: {json.dumps(job, indent=2, ensure_ascii=False)}")

    results = []

    pipeline.log_step_event(
        logger,
        'pipeline',
        'start',
        status='OK',
        total_steps=len(steps_sorted),
    )
    start_time = time.time()

    for index, name in enumerate(steps_sorted, start=1):
        if ctx.get('debug'):
            logger.info(f"[DEBUG] Queuing step {index}/{len(steps_sorted)}: {name}")

        pipeline.log_step_event(
            logger,
            name,
            'queued',
            status='OK',
            position=f"{index}/{len(steps_sorted)}",
        )
        result = _run_step(name, job, ctx)
        results.append(result)

        if ctx.get('debug'):
            logger.info(f"[DEBUG] Completed step {index}/{len(steps_sorted)}: {name} ({result['status']})")

    elapsed = round(time.time() - start_time, 1)

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
            io.log_json(ctx['logs'], kpi_status_obj)

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
        io.log_json(ctx['logs'], budget_status)

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
    common.add_argument('--dry-run', action='store_true')
    common.add_argument('--sample', type=int, default=0)
    common.add_argument('--time-budget-min', type=int, default=0)
    common.add_argument('--workers', type=int, default=8)
    common.add_argument('--json', action='store_true')
    common.add_argument('--resume', action='store_true')
    common.add_argument('--verbose', action='store_true', help='Enable verbose logging with all process details')
    common.add_argument('--debug', action='store_true', help='Enable debug mode with important debug information')
    common.add_argument('--max-ram-mb', type=int, default=0)

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
    batch_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging with all process details')
    batch_parser.add_argument('--debug', action='store_true', help='Enable debug mode with important debug information')
    batch_parser.add_argument('--max-ram-mb', type=int, default=0, help='Maximum RAM budget in MB (0 = unlimited)')
    batch_parser.add_argument('--continue-on-error', action='store_true', help='Continue processing other NAF codes if one fails')
    batch_parser.add_argument('--json', action='store_true', help='Output results in JSON format')

    sub.add_parser('resume', parents=[common])

    args = parser.parse_args(argv)
    logger = pipeline.configure_logging(args.verbose, getattr(args, 'debug', False))

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
