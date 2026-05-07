#!/usr/bin/env bash
# Run all librarian Stage 1 checks. Exit code is non-zero if any check
# failed; each check still runs even if a previous one failed, so users
# see every finding in one pass.

set -uo pipefail

cd "$(dirname "$0")/.."

overall=0

echo "→ Markdown link integrity (lychee)"
if ! uv run python scripts/librarian_check_links.py; then
    overall=1
fi
echo

echo "→ Code citations (file:line refs in inline code)"
if ! uv run python scripts/librarian_check_citations.py; then
    overall=1
fi
echo

if [ "$overall" -eq 0 ]; then
    echo "All Stage 1 checks passed."
else
    echo "One or more Stage 1 checks failed (see above)."
fi

exit "$overall"
