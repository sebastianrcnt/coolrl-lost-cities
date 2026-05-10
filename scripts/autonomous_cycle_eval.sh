#!/bin/bash
# Autonomous cycle eval helper.
# Usage: ./autonomous_cycle_eval.sh <run-prefix> [extra eval args...]
# Finds latest run matching prefix, runs eval --ckpt latest.pt with 30 games,
# and reports: timeouts, natural-end wins per opponent.

set -euo pipefail

PREFIX="${1:-}"
shift || true

if [ -z "$PREFIX" ]; then
    echo "usage: $0 <run-prefix> [extra eval args...]"
    exit 1
fi

RUN=$(ls -td runs/*${PREFIX}* 2>/dev/null | head -1)
if [ -z "$RUN" ]; then
    echo "no run matching ${PREFIX}" >&2
    exit 1
fi

CKPT="$RUN/latest.pt"
if [ ! -f "$CKPT" ]; then
    echo "no checkpoint at $CKPT" >&2
    exit 1
fi

echo "=== eval $CKPT ==="
uv run lost-cities-ismcts eval --ckpt "$CKPT" --games 30 --verbose "$@" 2>&1
