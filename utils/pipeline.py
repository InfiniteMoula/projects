
"""Shared helpers for pipeline steps resolution and logging."""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

LOGGER_NAME = "builder.pipeline"


def configure_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Configure and return the pipeline logger.

    The console handler is initialised lazily so repeated calls only
    adjust the level. The returned logger does not propagate to avoid
    duplicate entries when embedding in other apps.
    
    Args:
        verbose: Enable verbose (DEBUG level) logging with detailed information
        debug: Enable debug mode (INFO level) with important debug information
    """

    logger = logging.getLogger(LOGGER_NAME)
    
    # Determine log level based on flags
    if verbose:
        level = logging.DEBUG  # Most detailed logging
    elif debug:
        level = logging.INFO   # Important debug info
    else:
        level = logging.WARNING  # Only warnings and errors
    
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        # Enhanced formatter for debug/verbose modes
        if verbose or debug:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
            # Update formatter for existing handlers
            if verbose or debug:
                formatter = logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            else:
                formatter = logging.Formatter(
                    fmt="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            handler.setFormatter(formatter)
    return logger


def get_logger() -> logging.Logger:
    """Return the configured pipeline logger."""

    return logging.getLogger(LOGGER_NAME)


def resolve_target(target: str) -> Callable[..., object]:
    """Resolve "module:function" target strings to callables."""

    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"callable {target!r} not found") from exc


def resolve_step(step_name: str, registry: Mapping[str, str]) -> Callable[..., object]:
    """Resolve a step name using a registry mapping."""

    try:
        target = registry[step_name]
    except KeyError as exc:
        raise KeyError(f"unknown step {step_name!r}") from exc
    return resolve_target(target)


def log_step_event(
    logger: logging.Logger,
    step: str,
    event: str,
    *,
    status: Optional[str] = None,
    duration: Optional[float] = None,
    **extra: object,
) -> None:
    """Emit a structured log entry for a pipeline event."""

    payload: MutableMapping[str, object] = {
        "step": step,
        "event": event,
    }
    if status is not None:
        payload["status"] = status
    if duration is not None:
        payload["duration_s"] = duration
    payload.update(extra)
    message = " | ".join(f"{k}={v}" for k, v in payload.items())
    if status in (None, "OK", "SKIPPED"):
        level = logging.INFO
    elif status == "WARN":
        level = logging.WARNING
    else:
        level = logging.ERROR
    logger.log(level, message, extra={"step": step, "event": event})


def topo_sort(
    steps: Sequence[str],
    dependencies: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Return steps ordered according to their dependencies.

    Steps absent from the `steps` sequence are ignored. Missing dependencies
    that are not part of the requested `steps` are simply skipped so that
    optional edges do not break execution.
    """

    deps: MutableMapping[str, set[str]] = {}
    requested = []
    seen = set()
    for step in steps:
        if step in seen:
            continue
        seen.add(step)
        requested.append(step)
        if not dependencies:
            deps[step] = set()
        else:
            deps[step] = set(dependencies.get(step, ()))

    result: list[str] = []
    temp_mark: set[str] = set()

    def visit(node: str) -> None:
        if node in result:
            return
        if node in temp_mark:
            raise ValueError(f"cycle detected at step {node!r}")
        temp_mark.add(node)
        for dep in deps.get(node, ()):  # type: ignore[arg-type]
            if dep in deps:
                visit(dep)
        temp_mark.remove(node)
        if node not in result:
            result.append(node)

    for step in requested:
        visit(step)
    return result
