# projects_mcp_server.py
# MCP server exposing exploration tools limited to PROJECTS_ROOT.
# Installation:
#   pip install "mcp[cli]"
# Launch:
#   python projects_mcp_server.py
# Example MCP client config (JSON):
# {
#   "mcpServers": {
#     "projects-intel": {
#       "command": ["python", "projects_mcp_server.py"],
#       "env": {
#         "PROJECTS_ROOT": "/absolute/path/to/projects"
#       }
#     }
#   }
# }
"""
Production-ready MCP server (stdio transport) that exposes repository intelligence tools while
keeping every operation confined to the directory pointed to the PROJECTS_ROOT environment
variable. All file-system access is guarded so that the server never escapes the configured root.
Logging is routed to stderr; stdout is reserved for MCP transport messages.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "projects-intel"
PROJECTS_ROOT_ENV = "PROJECTS_ROOT"

IGNORE_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".cache",
    "__pycache__",
}

ALLOWED_READ_EXTENSIONS: set[str] = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".md",
    ".txt",
}

DEFAULT_GREP_GLOBS: list[str] = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.md"]
DEFAULT_REFERENCE_GLOBS: list[str] = ["**/*.py"]
DEFAULT_SNAPSHOT_PATH = Path("out/ai_context_snapshot.md")
MAX_PREVIEW_CHARS = 1200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger(__name__)


class PathResolutionError(ValueError):
    """Raised when a requested path is invalid or attempts to escape the root."""


def _load_projects_root() -> Path:
    """Read and validate the PROJECTS_ROOT environment variable."""
    raw_root = os.environ.get(PROJECTS_ROOT_ENV)
    if not raw_root:
        raise SystemExit(f"{PROJECTS_ROOT_ENV} is not set; export it to start the MCP server.")

    root_path = Path(raw_root).expanduser().resolve()
    if not root_path.is_dir():
        raise SystemExit(f"{PROJECTS_ROOT_ENV}={root_path} is not a directory.")
    return root_path


PROJECTS_ROOT: Path = _load_projects_root()


def _normalize_return_path(path: Path) -> str:
    """Return the path relative to PROJECTS_ROOT using POSIX separators."""
    relative = path.relative_to(PROJECTS_ROOT)
    return "." if relative == Path(".") else relative.as_posix()


def _is_ignored_dir(entry: Path) -> bool:
    """Check whether a directory should be skipped during traversal."""
    return entry.name in IGNORE_DIRS


def _safe_resolve(
    user_path: str | None, *, allow_nonexistent: bool = False, expect_dir: bool | None = None
) -> Path:
    """
    Resolve a user-supplied path relative to PROJECTS_ROOT while preventing escapes.

    Args:
        user_path: Relative path provided by a tool caller.
        allow_nonexistent: Allow returning paths that do not yet exist (for outputs).
        expect_dir: If True/False, enforce directory/file expectation when the path exists.
    """
    base = PROJECTS_ROOT
    if user_path is None or user_path == "":
        candidate = base
    else:
        supplied = Path(user_path)
        if supplied.is_absolute():
            raise PathResolutionError("Absolute paths are not allowed.")
        candidate = base / supplied

    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(base):
        raise PathResolutionError("Path escapes PROJECTS_ROOT.")

    if not allow_nonexistent and not resolved.exists():
        raise PathResolutionError(f"Path does not exist: {_normalize_return_path(resolved)}")

    if resolved.exists() and expect_dir is not None:
        if expect_dir and not resolved.is_dir():
            raise PathResolutionError("Expected a directory path.")
        if not expect_dir and not resolved.is_file():
            raise PathResolutionError("Expected a file path.")

    return resolved


def _iter_files_by_globs(root: Path, patterns: Iterable[str]) -> Iterator[Path]:
    """Yield files under root matching given glob patterns, respecting ignore rules."""
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            if path.is_symlink():
                continue
            if any(part in IGNORE_DIRS for part in path.relative_to(root).parts):
                continue
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(root):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _read_file_lines(path: Path) -> Iterator[str]:
    """Safely read a text file line by line with UTF-8 fallback decoding."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def _collect_dir_tree(
    path: Path, *, max_depth: int, include_files: bool, depth: int = 0
) -> dict[str, Any]:
    """Recursively collect a directory tree capped by max_depth."""
    node: dict[str, Any] = {"path": _normalize_return_path(path), "type": "dir", "children": []}
    if depth >= max_depth:
        return node

    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        LOGGER.warning("Permission denied while traversing %s", path)
        return node

    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if _is_ignored_dir(entry):
                continue
            node["children"].append(
                _collect_dir_tree(
                    entry, max_depth=max_depth, include_files=include_files, depth=depth + 1
                )
            )
        elif include_files:
            node["children"].append(
                {
                    "path": _normalize_return_path(entry),
                    "type": "file",
                    "children": [],
                }
            )
    return node


def _is_allowed_extension(path: Path) -> bool:
    """Check whether a path has an allowed extension for read operations."""
    return path.suffix.lower() in ALLOWED_READ_EXTENSIONS


def _classify_reference(symbol: str, line: str) -> str:
    """Heuristically classify the reference type for a symbol."""
    escaped = re.escape(symbol)
    if re.search(rf"\bfrom\s+[.\w]+\s+import\s+.*\b{escaped}\b", line) or re.search(
        rf"\bimport\s+.*\b{escaped}\b", line
    ):
        return "import"
    if re.search(rf"\b{escaped}\s*=", line):
        return "assignment"
    if re.search(rf"\b{escaped}\s*\(", line):
        return "call"
    return "unknown"


def _extract_files_scanned(text: str) -> int | None:
    """Best-effort extraction of a `files scanned` count from process output."""
    match = re.search(r"files?\s+scanned[:=\s]+(\d+)", text, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _read_preview(path: Path) -> str:
    """Read up to MAX_PREVIEW_CHARS from a text file."""
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[:MAX_PREVIEW_CHARS]


mcp = FastMCP(
    SERVER_NAME,
    instructions="Repository intelligence server restricted to PROJECTS_ROOT.",
    log_level="INFO",
)


@mcp.tool()
def list_repo_map(
    root: str | None = None, max_depth: int = 4, include_files: bool = True
) -> dict[str, Any]:
    """
    List the repository structure beneath PROJECTS_ROOT or a relative sub-directory.

    Args:
        root: Optional relative sub-directory to start from.
        max_depth: Maximum traversal depth (directories only).
        include_files: When False, only include directories in the map.
    """
    try:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        base = _safe_resolve(root, expect_dir=True)
        tree = _collect_dir_tree(base, max_depth=max_depth, include_files=include_files)
        return tree
    except Exception as exc:
        LOGGER.exception("list_repo_map failed")
        return {"error": str(exc)}


@mcp.tool()
def read_code(path: str, max_lines: int = 800) -> dict[str, Any]:
    """
    Read a text file under PROJECTS_ROOT with a line-capped snippet.

    Args:
        path: Relative path to the file.
        max_lines: Number of lines to include in the snippet (max).
    """
    try:
        if max_lines <= 0:
            raise ValueError("max_lines must be positive.")

        target = _safe_resolve(path, expect_dir=False)
        if not _is_allowed_extension(target):
            raise PathResolutionError(f"Extension not allowed for reading: {target.suffix}")

        total_lines = 0
        snippet_lines: list[str] = []
        for line in _read_file_lines(target):
            if total_lines < max_lines:
                snippet_lines.append(line)
            total_lines += 1

        snippet = "".join(snippet_lines)
        return {
            "path_norm": _normalize_return_path(target),
            "total_lines": total_lines,
            "snippet": snippet,
        }
    except Exception as exc:
        LOGGER.exception("read_code failed")
        return {"error": str(exc)}


@mcp.tool()
def grep_code(
    pattern: str, globs: list[str] | None = None, max_matches: int = 200
) -> dict[str, Any]:
    """
    Search for a regex pattern within repository text files.

    Args:
        pattern: Regular expression pattern to search.
        globs: Optional glob patterns to limit files.
        max_matches: Maximum number of matches to return.
    """
    if max_matches <= 0:
        return {"error": "max_matches must be positive."}

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}"}

    globs = globs or DEFAULT_GREP_GLOBS
    matches: list[dict[str, Any]] = []

    try:
        for file_path in _iter_files_by_globs(PROJECTS_ROOT, globs):
            if not _is_allowed_extension(file_path):
                continue
            for line_number, line in enumerate(_read_file_lines(file_path), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": _normalize_return_path(file_path),
                            "line_number": line_number,
                            "line_preview": line.rstrip("\n\r"),
                        }
                    )
                    if len(matches) >= max_matches:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}
    except Exception as exc:
        LOGGER.exception("grep_code failed")
        return {"error": str(exc)}


@mcp.tool()
def repo_stats() -> dict[str, Any]:
    """
    Compute statistics for the repository under PROJECTS_ROOT.

    Returns:
        Aggregate counts by extension, total Python LOC, and the top 10 longest Python files.
    """
    try:
        ext_counts: Counter[str] = Counter()
        python_line_total = 0
        python_lengths: list[tuple[int, Path]] = []

        for dirpath, dirnames, filenames in os.walk(PROJECTS_ROOT):
            dir_path = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in IGNORE_DIRS and not (dir_path / name).is_symlink()
            ]

            for filename in filenames:
                file_path = dir_path / filename
                if file_path.is_symlink():
                    continue
                try:
                    relative_parts = file_path.relative_to(PROJECTS_ROOT).parts
                except ValueError:
                    continue
                if any(part in IGNORE_DIRS for part in relative_parts):
                    continue

                suffix = file_path.suffix.lower()
                ext_counts[suffix] += 1

                if suffix == ".py":
                    line_count = 0
                    for _ in _read_file_lines(file_path):
                        line_count += 1
                    python_line_total += line_count
                    python_lengths.append((line_count, file_path))

        python_top = sorted(python_lengths, key=lambda item: item[0], reverse=True)[:10]
        return {
            "files_by_extension": dict(sorted(ext_counts.items(), key=lambda item: item[0])),
            "python": {
                "total_lines": python_line_total,
                "top_longest": [
                    {"path": _normalize_return_path(path), "lines": count}
                    for count, path in python_top
                ],
            },
        }
    except Exception as exc:
        LOGGER.exception("repo_stats failed")
        return {"error": str(exc)}


@mcp.tool()
def run_ai_context_snapshot(output_path: str | None = None) -> dict[str, Any]:
    """
    Run ai_context.py (if present) to generate a Markdown snapshot.

    Args:
        output_path: Optional relative output path; defaults to out/ai_context_snapshot.md.
    """
    try:
        script_path = PROJECTS_ROOT / "ai_context.py"
        if not script_path.exists():
            return {"error": "ai_context.py not found under PROJECTS_ROOT."}

        target_output = (
            _safe_resolve(output_path, allow_nonexistent=True)
            if output_path
            else (PROJECTS_ROOT / DEFAULT_SNAPSHOT_PATH).resolve()
        )

        if not target_output.is_relative_to(PROJECTS_ROOT):
            raise PathResolutionError("Output path escapes PROJECTS_ROOT.")

        target_output.parent.mkdir(parents=True, exist_ok=True)

        command = [sys.executable, str(script_path), "--output", str(target_output)]
        LOGGER.info("Executing snapshot command: %s", json.dumps(command))
        process = subprocess.run(
            command,
            cwd=str(PROJECTS_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        stdout_tail = process.stdout[-1000:] if process.stdout else None
        stderr_tail = process.stderr[-1000:] if process.stderr else None

        if process.returncode != 0:
            return {
                "error": f"ai_context.py exited with code {process.returncode}",
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }

        if not target_output.exists():
            return {
                "error": "Snapshot script finished but output file is missing.",
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }

        files_scanned = _extract_files_scanned(process.stdout + process.stderr)
        preview = _read_preview(target_output)
        return {
            "output_path": _normalize_return_path(target_output),
            "files_scanned": files_scanned,
            "preview": preview,
        }
    except Exception as exc:
        LOGGER.exception("run_ai_context_snapshot failed")
        return {"error": str(exc)}


@mcp.tool()
def find_references(
    symbol: str, globs: list[str] | None = None, max_matches: int = 200
) -> dict[str, Any]:
    """
    Locate references to a Python symbol across repository files.

    Args:
        symbol: Symbol name to search for (word boundary matched).
        globs: Optional glob patterns; defaults to Python sources.
        max_matches: Maximum matches to return.
    """
    if max_matches <= 0:
        return {"error": "max_matches must be positive."}

    globs = globs or DEFAULT_REFERENCE_GLOBS
    symbol_regex = re.compile(rf"\b{re.escape(symbol)}\b")
    matches: list[dict[str, Any]] = []

    try:
        for file_path in _iter_files_by_globs(PROJECTS_ROOT, globs):
            if file_path.suffix.lower() != ".py":
                continue
            for line_number, line in enumerate(_read_file_lines(file_path), start=1):
                if not symbol_regex.search(line):
                    continue
                matches.append(
                    {
                        "path": _normalize_return_path(file_path),
                        "line_number": line_number,
                        "line_preview": line.rstrip("\n\r"),
                        "match_type": _classify_reference(symbol, line),
                    }
                )
                if len(matches) >= max_matches:
                    return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}
    except Exception as exc:
        LOGGER.exception("find_references failed")
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run("stdio")
