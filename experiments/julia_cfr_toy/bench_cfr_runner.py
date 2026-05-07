from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ITERATIONS = 100
TRAVERSALS_PER_ITER = 1000
EPSILON = 1e-9


def _build_extension() -> None:
    subprocess.run(
        [sys.executable, "bench_cfr_setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
        check=True,
    )


def _import_extension():
    sys.path.insert(0, str(ROOT))
    source_mtime = (ROOT / "bench_cfr.pyx").stat().st_mtime
    extensions = list(ROOT.glob("bench_cfr*.so"))
    if not extensions or max(path.stat().st_mtime for path in extensions) < source_mtime:
        _build_extension()
    try:
        return importlib.import_module("bench_cfr")
    except ImportError:
        _build_extension()
        importlib.invalidate_caches()
        return importlib.import_module("bench_cfr")


def _julia_executable() -> str:
    local = REPO_ROOT / "tools" / "julia" / "current" / "bin" / "julia"
    if local.exists():
        return str(local)
    return "julia"


def _run_julia() -> dict[str, Any]:
    proc = subprocess.run(
        [_julia_executable(), str(ROOT / "bench_cfr.jl"), "--json"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def _run_cython() -> dict[str, Any]:
    bench_cfr = _import_extension()
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = bench_cfr.run_benchmark()
    total_s = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result["total_s"] = total_s
    result["iter_ms"] = total_s * 1000.0 / ITERATIONS
    result["traversal_us"] = total_s * 1_000_000.0 / (ITERATIONS * TRAVERSALS_PER_ITER)
    result["alloc_mb"] = peak / 1024.0 / 1024.0
    result["gc_time_s"] = None
    result["gc_share"] = None
    return result


def _assert_equivalent(julia: dict[str, Any], cython: dict[str, Any]) -> None:
    diffs = [
        abs(float(j_value) - float(c_value))
        for j_value, c_value in zip(julia["root_regret"], cython["root_regret"], strict=True)
    ]
    max_diff = max(diffs)
    if max_diff > EPSILON:
        raise SystemExit(
            "root regret mismatch: "
            f"max_diff={max_diff:.3e} "
            f"julia={julia['root_regret']} cython={cython['root_regret']}"
        )


def _print_table(julia: dict[str, Any], cython: dict[str, Any]) -> None:
    ratio = julia["iter_ms"] / cython["iter_ms"]
    gc_share = julia["gc_share"] * 100.0
    print("Lang     iter mean (ms)  total (s)  alloc (MB)  gc time (s)  gc share")
    print(
        f"{'Julia':<8} {julia['iter_ms']:14.2f} {julia['total_s']:10.2f} "
        f"{julia['alloc_mb']:11.1f} {julia['gc_time_s']:12.2f} {gc_share:8.1f}%"
    )
    print(
        f"{'Cython':<8} {cython['iter_ms']:14.2f} {cython['total_s']:10.2f} "
        f"{cython['alloc_mb']:11.1f} {'n/a':>12} {'-':>8}"
    )
    print(f"{'ratio':<8} {ratio:13.2f}x {'-':>10} {'-':>11} {'-':>12} {'-':>8}")

    if ratio < 0.95:
        verdict = "Julia faster"
    elif ratio > 1.05:
        verdict = "Julia slower"
    else:
        verdict = "rough parity"
    gc_note = "GC negligible" if gc_share < 5.0 else "GC visible"
    print(f"Julia/Cython iter ratio {ratio:.2f}x — {verdict}, {gc_note} ({gc_share:.1f}% GC)")


def main() -> None:
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    julia = _run_julia()
    cython = _run_cython()
    _assert_equivalent(julia, cython)
    _print_table(julia, cython)


if __name__ == "__main__":
    main()
