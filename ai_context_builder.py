#!/usr/bin/env python3
"""
Generate a Markdown snapshot of a repository to give an AI a compact yet rich
context of the project: global stats, file index, key files, excerpts, and raw
metadata. The script is standalone (stdlib only) and intended to run from the
repo root.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

CODE_EXT = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".cs"}
CONFIG_EXT = {".yaml", ".yml", ".json", ".toml", ".ini"}
DOC_EXT = {".md", ".rst", ".txt"}

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".cache",
    "dist",
    "build",
    "node_modules",
    "out",
    "logs",
}

MAX_STATS_READ_BYTES = 2_000_000
MAX_EXCERPT_READ_BYTES = 1_000_000
HEAD_LINE_LIMIT = 80
TAIL_LINE_LIMIT = 40
LARGE_FILE_BYTES = 10_000_000


@dataclass
class FileStats:
    path: str
    rel_path: str
    ext: str
    size_bytes: int
    line_count: int
    category: str
    is_large: bool


@dataclass
class Excerpt:
    path: str
    rel_path: str
    category: str
    kind: str  # "head" or "tail"
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genere un snapshot Markdown du contexte d'un repository."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Chemin vers la racine du repo a scanner.",
    )
    parser.add_argument(
        "--output",
        default="docs/ai_context_snapshot.md",
        help="Chemin vers le fichier Markdown genere.",
    )
    parser.add_argument(
        "--max-excerpts",
        type=int,
        default=40,
        help="Nombre maximum de fichiers pour lesquels on extrait des extraits.",
    )
    parser.add_argument(
        "--max-index-rows",
        type=int,
        default=500,
        help="Nombre maximum de lignes dans la table d'index (0 = illimite).",
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        dest="ignore_dirs",
        default=[],
        help="Dossier supplementaire a ignorer (option repetable).",
    )
    return parser.parse_args()


def determine_category(ext: str) -> str:
    if ext in CODE_EXT:
        return "code"
    if ext in CONFIG_EXT:
        return "config"
    if ext in DOC_EXT:
        return "doc"
    return "other"


def read_text_if_small(path: Path, limit_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
        if size > limit_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def build_file_stats(path: Path, root: Path) -> FileStats | None:
    try:
        stat_result = path.stat()
    except OSError:
        return None

    size_bytes = stat_result.st_size
    ext = path.suffix.lower()
    category = determine_category(ext)
    is_large = size_bytes >= LARGE_FILE_BYTES

    content = None
    if size_bytes <= MAX_STATS_READ_BYTES:
        content = read_text_if_small(path, MAX_STATS_READ_BYTES)

    line_count = len(content.splitlines()) if content is not None else 0
    rel_path = path.relative_to(root).as_posix()

    return FileStats(
        path=str(path.resolve()),
        rel_path=rel_path,
        ext=ext,
        size_bytes=size_bytes,
        line_count=line_count,
        category=category,
        is_large=is_large,
    )


def scan_repository(root: Path, ignore_dirs: set[str]) -> List[FileStats]:
    stats: List[FileStats] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.is_symlink():
                continue
            file_stats = build_file_stats(file_path, root)
            if file_stats:
                stats.append(file_stats)
    return stats


def category_counts(stats: Sequence[FileStats]) -> Dict[str, int]:
    counts: Dict[str, int] = {"code": 0, "config": 0, "doc": 0, "other": 0}
    for item in stats:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def select_top_files(
    stats: Sequence[FileStats], category: str, limit: int
) -> List[FileStats]:
    filtered = [s for s in stats if s.category == category]
    filtered.sort(key=lambda s: (-s.line_count, s.rel_path))
    return filtered[:limit]


def select_for_excerpts(stats: Sequence[FileStats], max_excerpts: int) -> List[FileStats]:
    selected: List[FileStats] = []
    for category in ("code", "config", "doc"):
        if len(selected) >= max_excerpts:
            break
        candidates = [s for s in stats if s.category == category and not s.is_large]
        candidates.sort(key=lambda s: (-s.line_count, s.rel_path))
        for item in candidates:
            if len(selected) >= max_excerpts:
                break
            selected.append(item)
    return selected


def build_excerpts(selected: Sequence[FileStats]) -> List[Excerpt]:
    excerpts: List[Excerpt] = []
    for file_stat in selected:
        path = Path(file_stat.path)
        content = read_text_if_small(path, MAX_EXCERPT_READ_BYTES)
        if content is None:
            note = "Contenu non lu (fichier volumineux ou erreur de lecture)."
            excerpts.append(
                Excerpt(
                    path=file_stat.path,
                    rel_path=file_stat.rel_path,
                    category=file_stat.category,
                    kind="head",
                    content=note,
                )
            )
            continue

        lines = content.splitlines()
        head_content = "\n".join(lines[:HEAD_LINE_LIMIT])
        excerpts.append(
            Excerpt(
                path=file_stat.path,
                rel_path=file_stat.rel_path,
                category=file_stat.category,
                kind="head",
                content=head_content,
            )
        )

        if len(lines) > HEAD_LINE_LIMIT:
            tail_content = "\n".join(lines[-TAIL_LINE_LIMIT:])
            if tail_content != head_content:
                excerpts.append(
                    Excerpt(
                        path=file_stat.path,
                        rel_path=file_stat.rel_path,
                        category=file_stat.category,
                        kind="tail",
                        content=tail_content,
                    )
                )
    return excerpts


def format_top_list(files: Sequence[FileStats]) -> List[str]:
    if not files:
        return ["- (aucun fichier)"]
    lines = []
    for item in files:
        size_kb = item.size_bytes / 1024
        lines.append(
            f"- `{item.rel_path}` - {item.line_count} lignes, {size_kb:.1f} Ko"
        )
    return lines


def build_index_table(stats: Sequence[FileStats], max_index_rows: int) -> List[str]:
    lines = [
        "| Categorie | Fichier | Lignes | Taille (Ko) | Extension | Gros fichier (oui/non) |",
        "|-----------|---------|--------|-------------|-----------|------------------------|",
    ]
    sorted_stats = sorted(stats, key=lambda s: (s.category, s.rel_path))
    truncated = False
    if max_index_rows and max_index_rows > 0 and len(sorted_stats) > max_index_rows:
        sorted_stats = sorted_stats[:max_index_rows]
        truncated = True
    for item in sorted_stats:
        size_kb = item.size_bytes / 1024
        ext_display = item.ext if item.ext else "(aucune)"
        large_display = "oui" if item.is_large else "non"
        lines.append(
            f"| {item.category} | `{item.rel_path}` | {item.line_count} | {size_kb:.1f} | {ext_display} | {large_display} |"
        )
    if truncated:
        lines.append("| ... | ... | ... | ... | ... | ... |")
    return lines


def excerpts_by_file(excerpts: Sequence[Excerpt]) -> Dict[str, List[Excerpt]]:
    grouped: Dict[str, List[Excerpt]] = {}
    for ex in excerpts:
        grouped.setdefault(ex.rel_path, []).append(ex)
    return grouped


def build_markdown(
    root: Path,
    stats: Sequence[FileStats],
    category_summary: Dict[str, int],
    top_code: Sequence[FileStats],
    top_config: Sequence[FileStats],
    top_doc: Sequence[FileStats],
    excerpts: Sequence[Excerpt],
    selected_for_excerpts: Sequence[FileStats],
    max_index_rows: int,
) -> str:
    total_files = len(stats)
    total_lines = sum(item.line_count for item in stats)

    lines: List[str] = []
    lines.append("# AI Context Snapshot")
    lines.append("")
    lines.append("## 1. Vue globale du repo")
    lines.append("")
    lines.append(f"- Nombre total de fichiers scannes : **{total_files}**")
    lines.append(f"- Nombre total de lignes (approx.) : **{total_lines}**")
    lines.append("")
    lines.append("### Repartition par categorie")
    for cat in ("code", "config", "doc", "other"):
        lines.append(f"- {cat} : **{category_summary.get(cat, 0)}** fichiers")
    lines.append("")
    lines.append("## 2. Index des fichiers")
    lines.append("")
    lines.extend(build_index_table(stats, max_index_rows))
    lines.append("")
    lines.append("## 3. Fichiers de code principaux")
    lines.append("")
    lines.extend(format_top_list(top_code))
    lines.append("")
    lines.append("## 4. Fichiers de configuration principaux")
    lines.append("")
    lines.extend(format_top_list(top_config))
    lines.append("")
    lines.append("## 4.b Fichiers de documentation principaux")
    lines.append("")
    lines.extend(format_top_list(top_doc))
    lines.append("")
    lines.append("## 5. Extraits de fichiers importants")
    lines.append("")
    lines.append(
        "> Remarque : ces extraits servent a donner a l'IA un apercu de la structure"
    )
    lines.append("> et des patterns du projet (signatures, docstrings, etc.).")
    lines.append("")

    grouped_excerpts = excerpts_by_file(excerpts)
    for fs in selected_for_excerpts:
        rel_path = fs.rel_path
        if rel_path not in grouped_excerpts:
            continue
        lines.append(f"### `{rel_path}`")
        file_excerpts = grouped_excerpts[rel_path]
        for ex in file_excerpts:
            label = "Debut du fichier" if ex.kind == "head" else "Fin du fichier"
            lines.append("")
            lines.append(f"#### {label}")
            lines.append("")
            lines.append("```")
            lines.append(ex.content)
            lines.append("```")
        lines.append("")

    meta = {
        "root": str(root),
        "total_files": total_files,
        "total_lines": total_lines,
        "category_counts": category_summary,
        "stats": [asdict(item) for item in stats],
        "selected_for_excerpts": [asdict(item) for item in selected_for_excerpts],
    }

    lines.append("## 6. Meta (brut)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Chemin de root invalide : {root}")

    ignore_dirs = set(DEFAULT_IGNORE_DIRS)
    ignore_dirs.update(args.ignore_dirs or [])

    stats = scan_repository(root, ignore_dirs)
    cat_counts = category_counts(stats)

    top_code = select_top_files(stats, "code", 20)
    top_config = select_top_files(stats, "config", 15)
    top_doc = select_top_files(stats, "doc", 15)
    selected = select_for_excerpts(stats, args.max_excerpts)
    excerpts = build_excerpts(selected)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        root=root,
        stats=stats,
        category_summary=cat_counts,
        top_code=top_code,
        top_config=top_config,
        top_doc=top_doc,
        excerpts=excerpts,
        selected_for_excerpts=selected,
        max_index_rows=args.max_index_rows,
    )
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Fichiers trouves : {len(stats)}")
    print(f"Fichiers avec stats : {len(stats)}")
    print(f"Fichiers selectionnes pour extraits : {len(selected)}")
    print(f"Markdown genere : {output_path.resolve()}")


if __name__ == "__main__":
    main()
