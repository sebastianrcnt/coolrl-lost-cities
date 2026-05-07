"""MEMORY.md drift check for librarian Stage 1.

Validates the user-memory directory:
- Each MEMORY.md index line points to a real memory file.
- Each memory file has YAML frontmatter with required fields
  (`name`, `description`, `type`) and a valid `type`.
- Reports orphan memory files (exist on disk but missing from index).

Memory dir is derived from the repo root, mirroring how Claude Code
resolves project-scoped memory paths. If the dir doesn't exist (fresh
checkout), the check exits 0 silently.

Usage:
    uv run python scripts/librarian_check_memory_drift.py
"""

from __future__ import annotations

import re
from pathlib import Path

VALID_TYPES = {"user", "feedback", "project", "reference"}
REQUIRED_FIELDS = ("name", "description", "type")

INDEX_LINE = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)")
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _memory_dir(root: Path) -> Path:
    project_slug = str(root).replace("/", "-")
    return Path.home() / ".claude" / "projects" / project_slug / "memory"


def _parse_frontmatter(body: str) -> dict[str, str] | None:
    m = FRONTMATTER.match(body)
    if not m:
        return None
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def main() -> int:
    root = _repo_root()
    mem_dir = _memory_dir(root)
    index_file = mem_dir / "MEMORY.md"
    if not index_file.is_file():
        print(f"MEMORY.md not found at {mem_dir}; skipping memory check.")
        return 0

    findings: list[str] = []
    referenced: set[str] = set()

    for raw in index_file.read_text(encoding="utf-8").splitlines():
        m = INDEX_LINE.match(raw)
        if not m:
            continue
        target = m.group(2).strip()
        referenced.add(target)
        target_path = mem_dir / target
        if not target_path.is_file():
            findings.append(f"index points to missing file: {target}")
            continue
        body = target_path.read_text(encoding="utf-8")
        fields = _parse_frontmatter(body)
        if fields is None:
            findings.append(f"{target}: missing or malformed frontmatter")
            continue
        for key in REQUIRED_FIELDS:
            if key not in fields:
                findings.append(f"{target}: frontmatter missing `{key}`")
        if "type" in fields and fields["type"] not in VALID_TYPES:
            findings.append(
                f"{target}: invalid type `{fields['type']}` (allowed: {sorted(VALID_TYPES)})"
            )

    on_disk = {p.name for p in mem_dir.glob("*.md") if p.name != "MEMORY.md"}
    for orphan in sorted(on_disk - referenced):
        findings.append(f"orphan memory file (not in MEMORY.md index): {orphan}")

    if not findings:
        print("MEMORY.md and memory files consistent.")
        return 0

    print("MEMORY.md drift findings:")
    for entry in findings:
        print(f"  {entry}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
