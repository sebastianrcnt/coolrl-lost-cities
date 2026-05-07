"""Stage 2 survey mode: promote-scan every archive without a research counterpart.

Walks `docs/archive/*.md`, filters out entries that already have a
`docs/research/<stem>.md` (after stripping any -YYYY-MM-DD suffix),
and dispatches each remaining archive to the configured LLM via the
same prompt assembly as `scripts/librarian_promote.py`. Outputs
land under `runs/tmp/librarian-survey-<timestamp>/`, one file per
archive.

Per archive output:
- `<stem>.md` — markdown draft, when the LLM judged the entry
  promotable (ready to copy into `docs/research/`).
- `<stem>.SKIP.txt` — the LLM's one-line reason, when it judged
  the entry non-promotable.
- `<stem>.ERROR.txt` — captured stderr, when the CLI itself failed.

Sequential; one LLM call per archive.

Usage:
    uv run python scripts/librarian_survey.py
    LIBRARIAN_LLM=gemini uv run python scripts/librarian_survey.py
    uv run python scripts/librarian_survey.py --dry-run    # list candidates only
    uv run python scripts/librarian_survey.py --max 3      # limit to first 3 candidates
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LLM_COMMANDS = {
    "claude": ["claude", "-p"],
    "codex": ["codex", "exec"],
    "gemini": ["gemini", "-p", ""],
}

DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml을 찾을 수 없어 repository root를 판정할 수 없습니다.")


def _assemble_prompt(
    system_prompt: str,
    rel_archive: Path,
    archive_body: str,
    rel_target: Path,
) -> str:
    return (
        f"{system_prompt}\n\n"
        "---\n\n"
        "Task: Draft a `docs/research/` note from the archive entry below.\n"
        "Follow the rules in your system prompt above (style template, "
        "`Last verified:` and `Source:` headers, `file:line` citations "
        "verified against the current tree, ~1 page, prose over bullet "
        "soup).\n\n"
        f"**Source archive:** `{rel_archive}`\n"
        f"**Suggested target filename:** `{rel_target}`\n\n"
        "If the archive does not contain a durable conclusion (e.g. it "
        "is a one-off bench result with no general lesson), respond "
        "with a single line `SKIP: <reason>` instead of a draft.\n\n"
        "Output: the markdown content of the new file only. No "
        "preamble, no code fences around the whole thing, no "
        "explanation after. Begin with the H1 header line.\n\n"
        "---\n\n"
        "Archive body:\n\n"
        f"{archive_body}\n"
    )


def _has_counterpart(archive_stem_no_date: str, research_stems: set[str]) -> bool:
    """True if a research note already covers this archive entry.

    Checks exact match plus tail match: archive `deep-cfr-foo-bar`
    is considered covered if research `foo-bar` exists, since some
    research notes intentionally drop a domain prefix.
    """
    if archive_stem_no_date in research_stems:
        return True
    parts = archive_stem_no_date.split("-")
    for i in range(1, len(parts)):
        tail = "-".join(parts[i:])
        if tail in research_stems:
            return True
    return False


def _candidates(root: Path) -> list[tuple[Path, Path]]:
    """Return (archive_path, suggested_research_target) for archives lacking a counterpart."""
    archive_dir = root / "docs" / "archive"
    research_dir = root / "docs" / "research"
    research_stems = {p.stem for p in research_dir.glob("*.md")}
    out: list[tuple[Path, Path]] = []
    for path in sorted(archive_dir.glob("*.md")):
        stem = DATE_SUFFIX.sub("", path.stem)
        if _has_counterpart(stem, research_stems):
            continue
        target = research_dir / f"{stem}.md"
        out.append((path, target))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate archives and their suggested research targets; do not call the LLM.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Process at most N archives (testing/cost guard).",
    )
    args = parser.parse_args()

    root = _repo_root()
    candidates = _candidates(root)
    if args.max is not None:
        candidates = candidates[: args.max]

    if not candidates:
        print("No archive entries without a research counterpart. Nothing to do.")
        return 0

    if args.dry_run:
        print(f"{len(candidates)} archive entries lack a research counterpart:")
        for archive, target in candidates:
            print(f"  {archive.relative_to(root)}  →  {target.relative_to(root)}")
        return 0

    backend = os.environ.get("LIBRARIAN_LLM", "claude").lower()
    cmd = LLM_COMMANDS.get(backend)
    if cmd is None:
        print(
            f"Unknown LIBRARIAN_LLM={backend}; supported: {sorted(LLM_COMMANDS)}",
            file=sys.stderr,
        )
        return 1

    system_prompt = (root / "scripts" / "librarian-prompt.md").read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = root / "runs" / "tmp" / f"librarian-survey-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    drafts = 0
    skips = 0
    errors = 0
    for idx, (archive, target) in enumerate(candidates, 1):
        rel_archive = archive.relative_to(root)
        rel_target = target.relative_to(root)
        print(f"[{idx}/{len(candidates)}] {rel_archive} → {backend}", file=sys.stderr)

        archive_body = archive.read_text(encoding="utf-8")
        prompt = _assemble_prompt(system_prompt, rel_archive, archive_body, rel_target)

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )
        stem_out = DATE_SUFFIX.sub("", archive.stem)
        if result.returncode != 0:
            (out_dir / f"{stem_out}.ERROR.txt").write_text(result.stderr, encoding="utf-8")
            errors += 1
            print(f"    ERROR (exit {result.returncode})", file=sys.stderr)
            continue

        output = result.stdout.strip()
        if output.lstrip().upper().startswith("SKIP"):
            (out_dir / f"{stem_out}.SKIP.txt").write_text(output, encoding="utf-8")
            skips += 1
            print(f"    SKIP ({output[:80]})", file=sys.stderr)
        else:
            (out_dir / f"{stem_out}.md").write_text(output, encoding="utf-8")
            drafts += 1
            print(f"    draft ({len(output)} chars)", file=sys.stderr)

    print()
    print(f"Survey complete: {drafts} drafts, {skips} skips, {errors} errors.")
    print(f"Output: {out_dir.relative_to(root)}")
    if drafts:
        print()
        print("Next: review drafts, then for each acceptable one:")
        print(f"  cp {out_dir.relative_to(root)}/<name>.md docs/research/<name>.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
