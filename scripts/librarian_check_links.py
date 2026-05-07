"""Lychee wrapper for librarian Stage 1 link integrity checks.

Ported from ../coolrl/src/coolrl/dev/check_doc_links.py. Walks the repo,
collects markdown files, and runs `lychee --offline` on them. Exit code
mirrors lychee's: 0 if all links resolve, non-zero otherwise.

Usage:
    uv run python scripts/librarian_check_links.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

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


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if not _is_excluded(rel):
            files.append(path)
    return sorted(set(files))


def main() -> int:
    lychee = shutil.which("lychee")
    if lychee is None:
        print(
            "lychee 실행 파일을 찾을 수 없습니다. "
            "`cargo install lychee` 또는 공식 설치 방법으로 lychee를 먼저 설치하세요.",
            file=sys.stderr,
        )
        return 127

    root = _repo_root()
    files = _markdown_files(root)
    if not files:
        print("검사할 Markdown 파일이 없습니다.", file=sys.stderr)
        return 1

    command = [
        lychee,
        "--offline",
        "--root-dir",
        str(root),
        *[str(path.relative_to(root)) for path in files],
    ]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
