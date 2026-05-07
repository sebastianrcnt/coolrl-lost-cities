# Plan: Vendor-Agnostic Librarian

**Status:** Design phase. AGENTS.md "Docs & Experiment Workflow" section
landed in commit `09d5815` (2026-05-07). Shell script and prompt-file
move not yet started.
**Owner:** operator-driven; Claude/Codex/Gemini may execute parts.
**Background:** A `librarian` subagent at `.claude/agents/librarian.md`
already drafts research notes and surveys docs, but it is Claude-only,
read-mostly, and cannot be triggered periodically. Doc placement rules
also lived inside that prompt instead of AGENTS.md, so non-librarian
agents never saw them.

## Goal

Split documentation hygiene into two layers:

1. **Authoring rules in AGENTS.md** — every agent reads these on every
   turn, so docs land in the right place at write time.
2. **`scripts/librarian.sh`** — periodic, vendor-agnostic, never
   auto-applies. Catches drift that Layer 1 missed.

Layer 1 already exists (commit `09d5815`). This plan covers Layer 2.

## Non-Goals

- Replacing the existing `librarian.md` prompt content. The note-drafting
  prompt is reused as a Stage 2 backend; only its location moves.
- Modifying `docs/archive/` or `runs/archive/`. Read-only forever.
- Editing code, configs, or running training/benchmarks from the
  librarian. Doc/memory work only.
- Any `auto-apply` mode. Librarian only proposes; humans (or a follow-up
  PR) apply.

## Architecture

Three stages, run in order. Each stage is independently invocable for
debugging.

### Stage 1 — Deterministic lint (no LLM)

Pure shell + `rg`/`find`/small Python helpers + [`lychee`](https://github.com/lycheeverse/lychee).
Output: a JSON report at `runs/tmp/librarian-<timestamp>.json`. Checks:

- **Markdown link integrity** (lychee): every `[text](path)` link in
  `docs/**` and `README.md` resolves; anchors point to real headers.
  Run `lychee --offline --root-dir . docs/**/*.md`. Precedent:
  `~/dev/coolrl/src/coolrl/dev/check_doc_links.py` wraps the same call
  for the sibling repo. We can lift that wrapper as-is.
- **Code-docs parity** (custom; lychee does not cover this): every
  `path/to/file.py:NN` citation in `docs/**` resolves (file exists,
  line within range). These are inline prose, not markdown links, so
  lychee ignores them. Short Python helper required.
- **Stale plans**: `docs/plans/*.md` with mtime > N days and no recent
  git commit referencing them.
- **Promotable archive**: `docs/archive/<name>-*.md` with no
  `docs/research/<name>.md` counterpart, where the archive body
  contains durable-conclusion language.
- **MEMORY.md drift**: index lines in `~/.claude/projects/.../MEMORY.md`
  that disagree with the target file's `description:` frontmatter.
- **Duplicate prose**: pairs of docs with high text overlap (e.g., a
  research note that copies an archive body instead of linking it).
- **Oversize**: files past the 500-line soft cap in AGENTS.md.

No LLM calls in Stage 1. Cheap to run frequently.

### Stage 2 — LLM judgment (vendor-agnostic)

Reads the Stage 1 report and the relevant doc bodies, dispatches to an
LLM CLI selected by env var:

```bash
LIBRARIAN_LLM=claude   # claude code
LIBRARIAN_LLM=codex    # codex cli
LIBRARIAN_LLM=gemini   # gemini cli
```

The system prompt is loaded from `scripts/librarian-prompt.md` (moved
from `.claude/agents/librarian.md`; same content). LLM produces:

- Research-note drafts for promotable archive entries.
- MEMORY.md drift fixups (one-line diffs).
- Duplicate-doc merge proposals.

Output format: a unified diff + a short rationale per change. Never
written to disk by the LLM directly — emitted as a patch file under
`runs/tmp/librarian-<timestamp>.patch`.

### Stage 3 — Dry-run apply (default) / human apply

Default: print the patch and exit. With `--apply`: `git apply` the patch
(still requires the human to commit). Conflicts surface as standard
patch failures — operator resolves manually.

`docs/archive/` and `runs/archive/` are filtered out of any patch
target before apply.

## Concurrency Policy

Librarian is **never invoked from within an active agent session**. It
runs on demand by the operator (or via cron / post-commit hook). Because
Stage 3 is propose-only by default, two parties editing the same file
cannot corrupt each other — git's 3-way merge handles overlap when the
operator applies the patch.

## Open Questions

- Cron cadence? (start with manual-only; add cron once Stage 1 is
  stable)
- "Durable-conclusion language" detection in Stage 1 — keyword heuristic
  vs. defer to Stage 2 entirely. Default to deferring; Stage 1 just
  flags every archive without a research counterpart.
- Where the Stage 1 ignore-list lives once false positives accumulate.
  Tentatively `scripts/librarian-ignore.txt` with one rg-style pattern
  per line.

## Progress

- ✅ AGENTS.md "Docs & Experiment Workflow" section landed
  (commit `09d5815`, 2026-05-07).
- ✅ Plan drafted at `docs/plans/librarian.md` (this file).
- ✅ Prompt moved: `.claude/agents/librarian.md` →
  `scripts/librarian-prompt.md`. Claude-specific subagent registration
  removed.
- ✅ Stage 1, piece 1: `scripts/librarian_check_links.py` (lychee
  wrapper). Caught one stale README link on first run (commit
  `b9bbb4f`).
- ✅ Stage 1, piece 2: `scripts/librarian_check_citations.py` (custom
  `file:line` citation checker over inline-code spans). Skips
  `docs/archive/` and `docs/plans/archive/`. Ignore list at
  `scripts/librarian-ignore.txt` for intentional future-tense
  references. Caught one real drift in
  `docs/research/optimization_sequencing.md` (path moved into
  `docs/plans/archive/`).
- ✅ Stage 1 orchestrator: `scripts/librarian.sh`. Runs every Stage 1
  check in order, aggregates exit code, prints findings inline. Single
  entry point for users and (future) cron.
- ✅ AGENTS.md mentions `scripts/librarian.sh` as the doc-lint entry
  point in "Notes For Future Agents" (commit `0b363b5`).
- ✅ Stage 1, piece 3: `scripts/librarian_check_oversize.py`. Flags
  any non-archive markdown file over the 500-line soft cap declared
  in AGENTS.md. Caught one real finding on first run:
  `docs/performance.md` at 914 lines — split into sub-topics deferred
  as a separate task.

## Stage 1 Remaining Checks

- Stale plans (mtime + git-log staleness heuristic).
- Promotable archive entries (deferred to Stage 2 — heuristic vs LLM
  judgment is the open question).
- MEMORY.md drift (index lines vs target file `description:` frontmatter).
- Duplicate prose (high-overlap pairs across archive vs research).

## Next Concrete Step

Stale plan check (`scripts/librarian_check_stale_plans.py`). Walks
`docs/plans/*.md` and flags files whose mtime is older than N days
AND whose path hasn't appeared in `git log` over the same window.
This catches plans that drift out of mind without being archived.
