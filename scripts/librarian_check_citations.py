"""Code citation checker for librarian Stage 1.

Walks markdown files, extracts file:line citations from inline code
spans (single backticks), and verifies each path exists and (if a line
number is given) is within range.

Inline code only — fenced code blocks are intentionally skipped to keep
false positives down (snippets often contain string literals like
"foo.py" that aren't real cross-references).

Citations to external repos (paths whose first segment isn't one of
this repo's tracked top-level dirs) are silently skipped.

Usage:
    uv run python scripts/librarian_check_citations.py
"""

from __future__ import annotations

import fnmatch
import re
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

ALLOWED_TOP = {"src", "docs", "tests", "configs", "scripts", "experiments"}

INLINE_CODE = re.compile(r"`([^`\n]+)`")
PATH_REF = re.compile(
    r"([\w.-]+(?:/[\w.-]+)+\.(?:pyx|pxd|py|toml|yaml|yml|json|md|sh|rs|txt|c|h))"
    r"(?::(\d+))?"
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


EXCLUDED_DOC_PREFIXES = (
    Path("docs/archive"),
    Path("docs/plans/archive"),
)


def _is_archive_doc(rel: Path) -> bool:
    return any(prefix in rel.parents for prefix in EXCLUDED_DOC_PREFIXES)


def _markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if _is_excluded(rel) or _is_archive_doc(rel):
            continue
        files.append(path)
    return sorted(set(files))


def _load_ignore_patterns(root: Path) -> list[str]:
    ignore_file = root / "scripts" / "librarian-ignore.txt"
    if not ignore_file.is_file():
        return []
    patterns: list[str] = []
    for raw in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def _is_ignored(target: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(target, pat) for pat in patterns)


def _check_citation(root: Path, target: str, line_num: int | None) -> str | None:
    """Return error message if the citation is broken; None if OK or external."""
    first = target.split("/", 1)[0]
    if first not in ALLOWED_TOP:
        return None

    target_path = (root / target).resolve()
    try:
        target_path.relative_to(root.resolve())
    except ValueError:
        return None

    if not target_path.is_file():
        return f"file not found: {target}"

    if line_num is not None:
        with target_path.open(encoding="utf-8", errors="replace") as f:
            actual_lines = sum(1 for _ in f)
        if line_num > actual_lines:
            return f"line {line_num} out of range (file has {actual_lines} lines)"

    return None


def main() -> int:
    root = _repo_root()
    files = _markdown_files(root)
    if not files:
        print("검사할 Markdown 파일이 없습니다.", file=sys.stderr)
        return 1

    ignore_patterns = _load_ignore_patterns(root)

    errors: list[tuple[Path, int, str, int | None, str]] = []
    for doc in files:
        rel = doc.relative_to(root)
        with doc.open(encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, 1):
                for span in INLINE_CODE.finditer(line):
                    span_text = span.group(1)
                    if "://" in span_text:
                        continue
                    for match in PATH_REF.finditer(span_text):
                        target = match.group(1)
                        line_num_str = match.group(2)
                        line_num = int(line_num_str) if line_num_str else None
                        if _is_ignored(target, ignore_patterns):
                            continue
                        err = _check_citation(root, target, line_num)
                        if err:
                            errors.append((rel, line_idx, target, line_num, err))

    if not errors:
        print("All code citations resolve.")
        return 0

    for rel, doc_line, target, line_num, err in errors:
        cite = f"{target}:{line_num}" if line_num else target
        print(f"{rel}:{doc_line}: `{cite}` — {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
