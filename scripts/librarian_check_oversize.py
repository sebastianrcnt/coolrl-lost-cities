"""Oversize file check for librarian Stage 1.

Walks markdown files and flags any whose line count exceeds the
soft cap declared in AGENTS.md ("Docs & Experiment Workflow"
section). Skips read-only archive doc directories — splitting them
isn't an option.

Usage:
    uv run python scripts/librarian_check_oversize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SOFT_CAP_LINES = 500

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "runs",
    "target",
    "tools",
    "wheels",
}

EXCLUDED_DOC_PREFIXES = (
    Path("docs/archive"),
    Path("docs/plans/archive"),
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _is_excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    return any(prefix in rel.parents for prefix in EXCLUDED_DOC_PREFIXES)


def _markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if not _is_excluded(rel):
            files.append(path)
    return sorted(set(files))


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def main() -> int:
    root = _repo_root()
    files = _markdown_files(root)
    if not files:
        print("검사할 Markdown 파일이 없습니다.", file=sys.stderr)
        return 1

    over = [(path, _line_count(path)) for path in files]
    over = [(p, n) for p, n in over if n > SOFT_CAP_LINES]

    if not over:
        print(f"All markdown files under {SOFT_CAP_LINES}-line soft cap.")
        return 0

    over.sort(key=lambda item: -item[1])
    print(f"Files over {SOFT_CAP_LINES}-line soft cap:")
    for path, n in over:
        rel = path.relative_to(root)
        print(f"  {n:5d}  {rel}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
