"""Stale plan check for librarian Stage 1.

Walks docs/plans/*.md (top-level only — archive subdir is separate)
and flags plans whose last git commit is older than STALE_DAYS.
Plans should be either actively worked on or archived (moved to
docs/plans/archive/) — long-untouched files are usually drift.

Usage:
    uv run python scripts/librarian_check_stale_plans.py
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

STALE_DAYS = 60


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _last_commit_date(root: Path, rel: Path) -> str | None:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", str(rel)],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    out = result.stdout.strip()
    return out if out else None


def main() -> int:
    root = _repo_root()
    plans_dir = root / "docs" / "plans"
    if not plans_dir.is_dir():
        print("docs/plans not found; skipping stale-plan check.")
        return 0

    cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).date()

    stale: list[tuple[Path, str]] = []
    for path in sorted(plans_dir.glob("*.md")):
        rel = path.relative_to(root)
        last = _last_commit_date(root, rel)
        if last is None:
            stale.append((rel, "untracked"))
            continue
        try:
            last_date = datetime.strptime(last, "%Y-%m-%d").date()
        except ValueError:
            continue
        if last_date < cutoff:
            stale.append((rel, last))

    if not stale:
        print(f"No stale plans (all touched within {STALE_DAYS} days).")
        return 0

    print(f"Stale plan candidates (last commit > {STALE_DAYS} days ago):")
    for rel, last in stale:
        print(f"  {last}  {rel}")
    print()
    print("Action: archive (mv to docs/plans/archive/) or update if still active.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
