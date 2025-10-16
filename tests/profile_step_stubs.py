"""Stub pipeline steps for profile integration tests."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

EXECUTED_STEPS: List[str] = []


def reset() -> None:
    """Reset recorded step executions."""
    EXECUTED_STEPS.clear()


def _record_execution(step_name: str):
    """Create a stub runner that records execution order for *step_name*."""

    def _run(_cfg, ctx):
        outdir = Path(ctx.get("outdir") or ctx.get("outdir_path") or ".")
        outdir.mkdir(parents=True, exist_ok=True)
        EXECUTED_STEPS.append(step_name)
        ctx.setdefault("_executed_steps", []).append(step_name)
        if step_name == "finalize.premium_dataset":
            (outdir / "dataset.csv").write_text("id\n1\n")
            (outdir / "dataset.parquet").write_bytes(b"")
        return {"status": "OK", "step": step_name}

    _run.__name__ = f"run_{step_name.replace('.', '_')}"
    return _run


PROFILE_STEPS: Iterable[str] = [
    "dumps.collect",
    "api.collect",
    "normalize.standardize",
    "quality.checks",
    "quality.score",
    "package.export",
    "finalize.premium_dataset",
    "feeds.collect",
    "parse.jsonld",
    "enrich.address",
    "enrich.google_maps",
    "enrich.domain",
    "enrich.site",
    "enrich.dns",
    "enrich.email",
    "enrich.phone",
    "enrich.linkedin_clearbit_lite",
    "headless.collect",
    "pdf.collect",
    "parse.pdf",
    "parse.html",
]

STUB_FUNCTIONS: Dict[str, object] = {
    step: _record_execution(step) for step in PROFILE_STEPS
}

for step, fn in STUB_FUNCTIONS.items():
    globals()[fn.__name__] = fn


def build_registry() -> Dict[str, str]:
    """Return STEP_REGISTRY overrides for profile stub steps."""
    return {step: f"{__name__}:{fn.__name__}" for step, fn in STUB_FUNCTIONS.items()}


def executions() -> List[str]:
    """Return a copy of the executed steps log."""
    return list(EXECUTED_STEPS)

