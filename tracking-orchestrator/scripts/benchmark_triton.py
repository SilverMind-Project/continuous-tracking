"""Benchmark p50/p99 latency for all three Triton CTS models.

Measures wall-clock latency (client-side, including gRPC overhead) for each
model at batch sizes 1, 4, 8.  Run this after materialising model files
and starting Triton (docker compose up triton).

Usage:
    cd continuous-tracking
    uv run --extra triton tracking-orchestrator/scripts/benchmark_triton.py
    uv run --extra triton tracking-orchestrator/scripts/benchmark_triton.py \\
        --url localhost:8701 --warmup 20 --iters 200

Exit code 1 if any model exceeds its batch-8 p99 DoD target.

Performance targets (phase-0 section 0.31):
    person-detector  batch=8  p99 <= 12 ms  (RTX 4060-class GPU)
    reid-solider     batch=8  p99 <= 8 ms
    pose-rtmpose     batch=8  p99 <= 8 ms
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Model specs: (name, input_name, input_shape_per_sample, output_names)
# ---------------------------------------------------------------------------

MODEL_SPECS: list[tuple[str, str, tuple[int, ...], list[str]]] = [
    ("person-detector", "images", (3, 640, 640), ["output0"]),
    ("reid-solider", "input", (3, 256, 128), ["output"]),
    ("pose-rtmpose", "input", (3, 256, 192), ["simcc_x", "simcc_y"]),
]

BATCH_SIZES = [1, 4, 8]


@dataclass
class LatencyStats:
    model_name: str
    batch_size: int
    p50_ms: float
    p99_ms: float
    n_iters: int


async def bench_model(
    client: Any,
    model_name: str,
    input_name: str,
    sample_shape: tuple[int, ...],
    output_names: list[str],
    batch_size: int,
    warmup: int,
    iters: int,
) -> LatencyStats:
    """Benchmark one model at one batch size."""
    try:
        import tritonclient.grpc as tc  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install tritonclient: uv sync --extra triton") from exc

    batch_shape = (batch_size, *sample_shape)
    dummy = np.random.rand(*batch_shape).astype(np.float32)

    def build_request() -> tuple[list[Any], list[Any]]:
        inp: Any = tc.InferInput(input_name, list(batch_shape), "FP32")
        inp.set_data_from_numpy(dummy)
        outs = [tc.InferRequestedOutput(n) for n in output_names]
        return [inp], outs

    # Warmup
    for _ in range(warmup):
        ins, outs = build_request()
        await client.infer(model_name=model_name, inputs=ins, outputs=outs)

    # Measurement
    latencies: list[float] = []
    for _ in range(iters):
        ins, outs = build_request()
        t0 = time.perf_counter()
        await client.infer(model_name=model_name, inputs=ins, outputs=outs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies, dtype=np.float64)
    return LatencyStats(
        model_name=model_name,
        batch_size=batch_size,
        p50_ms=float(np.percentile(arr, 50)),
        p99_ms=float(np.percentile(arr, 99)),
        n_iters=iters,
    )


async def run(url: str, warmup: int, iters: int) -> list[LatencyStats]:
    try:
        import tritonclient.grpc.aio as aio  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install tritonclient: uv sync --extra triton") from exc

    results: list[LatencyStats] = []
    async with aio.InferenceServerClient(url=url, verbose=False) as client:
        # Readiness check
        for model_name, *_ in MODEL_SPECS:
            ready: bool = await client.is_model_ready(model_name)
            if not ready:
                print(
                    f"  [SKIP] {model_name} is not ready — load model files first.",
                    file=sys.stderr,
                )

        for model_name, input_name, sample_shape, output_names in MODEL_SPECS:
            for batch in BATCH_SIZES:
                print(f"  benchmarking {model_name} batch={batch} ...", end=" ", flush=True)
                try:
                    stat = await bench_model(
                        client,
                        model_name,
                        input_name,
                        sample_shape,
                        output_names,
                        batch,
                        warmup,
                        iters,
                    )
                    results.append(stat)
                    print(f"p50={stat.p50_ms:.1f}ms  p99={stat.p99_ms:.1f}ms")
                except Exception as exc:  # noqa: BLE001
                    # Benchmarking continues across models so one unavailable
                    # backend does not hide the latency for the rest.
                    print(f"ERROR: {exc}")

    return results


def _print_table(results: list[LatencyStats]) -> None:
    header = f"{'Model':<20} {'Batch':>5} {'p50 (ms)':>10} {'p99 (ms)':>10} {'Target':>10}"
    print("\n" + header)
    print("-" * len(header))

    targets: dict[tuple[str, int], float] = {
        ("person-detector", 8): 12.0,
        ("reid-solider", 8): 8.0,
        ("pose-rtmpose", 8): 8.0,
    }

    for r in results:
        target = targets.get((r.model_name, r.batch_size), 0.0)
        target_str = f"<={target:.0f}" if target else "—"
        flag = " ✗" if target and r.p99_ms > target else ""
        print(
            f"{r.model_name:<20} {r.batch_size:>5} "
            f"{r.p50_ms:>10.2f} {r.p99_ms:>10.2f} {target_str:>10}{flag}"
        )


def _check_dod(results: list[LatencyStats]) -> bool:
    """Return True if all DoD latency gates pass."""
    gates: dict[tuple[str, int], float] = {
        ("person-detector", 8): 12.0,
        ("reid-solider", 8): 8.0,
        ("pose-rtmpose", 8): 8.0,
    }
    passed = True
    for (model, batch), limit in gates.items():
        matching = [r for r in results if r.model_name == model and r.batch_size == batch]
        if not matching:
            print(f"  [MISSING] No result for {model} batch={batch}", file=sys.stderr)
            passed = False
            continue
        r = matching[0]
        if r.p99_ms > limit:
            print(
                f"  [FAIL] {model} batch={batch} p99={r.p99_ms:.1f}ms > {limit}ms DoD gate",
                file=sys.stderr,
            )
            passed = False
        else:
            print(f"  [PASS] {model} batch={batch} p99={r.p99_ms:.1f}ms <= {limit}ms")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8701", help="Triton gRPC URL")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations")
    args = parser.parse_args()

    print(f"Triton benchmark: {args.url}  warmup={args.warmup}  iters={args.iters}")
    results = asyncio.run(run(args.url, args.warmup, args.iters))

    _print_table(results)

    print("\nDoD gate check (phase-0 section 0.31):")
    if not _check_dod(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
