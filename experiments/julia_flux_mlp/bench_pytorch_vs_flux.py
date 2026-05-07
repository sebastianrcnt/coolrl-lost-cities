from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from coolrl_lost_cities.games.classic.deep_cfr.networks import DeepCFRMLP

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
BATCH_SIZES = (1, 64, 256)
INPUT_DIM = 365
HIDDEN_SIZE = 512
NUM_LAYERS = 3
OUTPUT_DIM = 22
WARMUP_ITERS = 10
TIMED_ITERS = 100
SEED = 20260507
MAX_OUTPUT_DIFF = 1e-4


def _julia_executable() -> str:
    local = REPO_ROOT / "tools" / "julia" / "current" / "bin" / "julia"
    if local.exists():
        return str(local)
    return "julia"


def _linear_layers(model: DeepCFRMLP) -> list[torch.nn.Linear]:
    return [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]


def _export_layers(model: DeepCFRMLP) -> list[dict[str, Any]]:
    layers = []
    for layer in _linear_layers(model):
        weight = layer.weight.detach().cpu().contiguous()
        bias = layer.bias.detach().cpu().contiguous()
        layers.append(
            {
                "in_dim": int(weight.shape[1]),
                "out_dim": int(weight.shape[0]),
                "weight": weight.flatten().tolist(),
                "bias": bias.flatten().tolist(),
            }
        )
    return layers


def _time_pytorch(model: DeepCFRMLP, x: torch.Tensor) -> float:
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            model(x)
        torch.cuda.synchronize()
    return float((time.perf_counter() - start) * 1000.0 / TIMED_ITERS)


def _build_payload_and_pytorch_results(payload_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch; aborting Criterion 4")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = (
        DeepCFRMLP(
            INPUT_DIM,
            OUTPUT_DIM,
            HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            activation="relu",
        )
        .eval()
        .cuda()
    )

    payload: dict[str, Any] = {
        "input_dim": INPUT_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "output_dim": OUTPUT_DIM,
        "activation": "relu",
        "layers": _export_layers(model),
        "batches": [],
    }
    pytorch_batches: dict[str, Any] = {}
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            x_cpu = torch.randn(batch_size, INPUT_DIM, generator=generator, dtype=torch.float32)
            x_gpu = x_cpu.cuda()
            y_cpu = model(x_gpu).detach().cpu().contiguous()
            forward_ms = _time_pytorch(model, x_gpu)
            payload["batches"].append(
                {
                    "batch_size": batch_size,
                    "input_dim": INPUT_DIM,
                    "input": x_cpu.contiguous().flatten().tolist(),
                    "expected_output": y_cpu.flatten().tolist(),
                }
            )
            pytorch_batches[str(batch_size)] = {
                "forward_ms": forward_ms,
                "us_per_state": forward_ms * 1000.0 / batch_size,
            }

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "lang": "PyTorch",
        "device": torch.cuda.get_device_name(0),
        "timed_iters": TIMED_ITERS,
        "warmup_iters": WARMUP_ITERS,
        "batches": pytorch_batches,
    }


def _run_flux(payload_path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            _julia_executable(),
            "--project=experiments/julia_flux_mlp",
            "experiments/julia_flux_mlp/bench_flux.jl",
            str(payload_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def _verdict(pytorch: dict[str, Any], flux: dict[str, Any]) -> tuple[str, dict[str, float]]:
    ratios = {}
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        ratio = flux["batches"][key]["us_per_state"] / pytorch["batches"][key]["us_per_state"]
        ratios[key] = ratio
    within_64 = 0.8 <= ratios["64"] <= 1.2
    within_1 = 0.7 <= ratios["1"] <= 1.3
    within_256 = 0.7 <= ratios["256"] <= 1.3
    return ("PASS" if within_64 and within_1 and within_256 else "FAIL"), ratios


def _print_table(
    pytorch: dict[str, Any], flux: dict[str, Any], ratios: dict[str, float], verdict: str
) -> None:
    print("Backend   batch  forward_ms  us/state  ratio vs PyTorch")
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        p = pytorch["batches"][key]
        f = flux["batches"][key]
        print(
            f"PyTorch   {batch_size:5d}  {p['forward_ms']:10.4f}  {p['us_per_state']:8.4f}  1.00x"
        )
        print(
            f"Flux      {batch_size:5d}  {f['forward_ms']:10.4f}  "
            f"{f['us_per_state']:8.4f}  {ratios[key]:.2f}x"
        )
    print(f"max output diff: {flux['max_abs_diff']:.3e}")
    print(f"Criterion 4 verdict: {verdict}")


def main() -> None:
    payload_path = REPO_ROOT / "runs" / "tmp" / "julia_flux_mlp_payload.json"
    pytorch = _build_payload_and_pytorch_results(payload_path)
    flux = _run_flux(payload_path)
    if float(flux["max_abs_diff"]) > MAX_OUTPUT_DIFF:
        raise SystemExit(f"Flux output mismatch: max_abs_diff={flux['max_abs_diff']:.3e}")
    verdict, ratios = _verdict(pytorch, flux)
    result = {"pytorch": pytorch, "flux": flux, "ratios": ratios, "verdict": verdict}
    result_path = ROOT / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(pytorch, flux, ratios, verdict)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
