"""Stage 2 LLM dispatcher: draft a research note from an archive entry.

Reads `docs/archive/<name>.md`, assembles a prompt by stitching
`scripts/librarian-prompt.md` (the system prompt) onto the archive
body, and dispatches to the LLM CLI selected by the LIBRARIAN_LLM
environment variable. The LLM's stdout is captured to
`runs/tmp/librarian-promote-<timestamp>-draft.md`. The dispatcher
never writes into `docs/research/` directly — the operator reviews
the draft and copies/edits it themselves.

Backends:
    LIBRARIAN_LLM=claude   (default; invokes `claude -p`)
    LIBRARIAN_LLM=codex    (invokes `codex exec`)
    LIBRARIAN_LLM=gemini   (invokes `gemini -p`)

Usage:
    uv run python scripts/librarian_promote.py docs/archive/foo.md
    uv run python scripts/librarian_promote.py docs/archive/foo.md --show-prompt
    LIBRARIAN_LLM=codex uv run python scripts/librarian_promote.py docs/archive/foo.md
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
    "gemini": ["gemini", "-p"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "archive_path",
        help="Path to a docs/archive/*.md entry to promote.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the assembled prompt to stdout and exit; do not call the LLM.",
    )
    args = parser.parse_args()

    root = _repo_root()
    archive = (root / args.archive_path).resolve()

    if not archive.is_file():
        print(f"Archive not found: {args.archive_path}", file=sys.stderr)
        return 1
    try:
        rel_archive = archive.relative_to(root)
    except ValueError:
        print(f"Archive must live under repo root: {archive}", file=sys.stderr)
        return 1
    if not str(rel_archive).startswith("docs/archive/"):
        print(
            f"Refusing: archive must live under docs/archive/: {rel_archive}",
            file=sys.stderr,
        )
        return 1

    stem = DATE_SUFFIX.sub("", archive.stem)
    target = root / "docs" / "research" / f"{stem}.md"
    rel_target = target.relative_to(root)

    if target.exists():
        print(
            f"Refusing: research counterpart already exists: {rel_target}",
            file=sys.stderr,
        )
        return 1

    system_prompt = (root / "scripts" / "librarian-prompt.md").read_text(encoding="utf-8")
    archive_body = archive.read_text(encoding="utf-8")
    prompt = _assemble_prompt(system_prompt, rel_archive, archive_body, rel_target)

    if args.show_prompt:
        sys.stdout.write(prompt)
        return 0

    backend = os.environ.get("LIBRARIAN_LLM", "claude").lower()
    cmd = LLM_COMMANDS.get(backend)
    if cmd is None:
        print(
            f"Unknown LIBRARIAN_LLM={backend}; supported: {sorted(LLM_COMMANDS)}",
            file=sys.stderr,
        )
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = root / "runs" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / f"librarian-promote-{timestamp}-prompt.md"
    draft_path = out_dir / f"librarian-promote-{timestamp}-draft.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    print(
        f"Dispatching to {backend} (prompt saved to {prompt_path.relative_to(root)})",
        file=sys.stderr,
    )
    result = subprocess.run(
        cmd + [prompt],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"LLM call failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    draft_path.write_text(result.stdout, encoding="utf-8")
    print(f"Draft written to: {draft_path.relative_to(root)}")
    print(f"Suggested target on accept: {rel_target}")
    print()
    print("Next: review the draft. To accept verbatim:")
    print(f"  cp {draft_path.relative_to(root)} {rel_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
