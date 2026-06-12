"""Equivalence gate: person-detector vs person-detector-dynamic.

Runs N representative frames through both models and asserts that detected
boxes agree within the same tolerances used for INT8 Jetson qualification:
  - recall/precision agreement at IoU 0.5 (no box present in one but not the other)
  - per-box confidence MAE <= 0.02

Requires a running Triton instance with both models loaded and a calibration
image set (private, outside git) specified via DETECTOR_CALIB_IMAGES_DIR.

Usage:
    TRITON_GRPC_URL=localhost:8701 \\
    DETECTOR_CALIB_IMAGES_DIR=/data/calib/frames \\
    python triton-models/scripts/verify_detector_equivalence.py

Make target (requires GPU + models; not part of make ci):
    make detector-equivalence

Exit codes:
    0 — all frames pass equivalence gate
    1 — gate failure (delta exceeds tolerance) or setup error
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Constants — match the INT8 qualification tolerances
# ---------------------------------------------------------------------------

_IOU_THRESHOLD = 0.5
_CONF_MAE_THRESHOLD = 0.02
_DEFAULT_N_FRAMES = 50
_DEFAULT_TRITON_URL = "localhost:8701"

_STATIC_MODEL = "person-detector"
_DYNAMIC_MODEL = "person-detector-dynamic"


def _iou(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    """IoU between two boxes [x1,y1,x2,y2] in the same pixel space."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _load_images(calib_dir: Path, n: int) -> list[npt.NDArray[np.uint8]]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        print(f"ERROR: no images found in {calib_dir}", file=sys.stderr)
        sys.exit(1)
    paths = paths[:n]
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"WARNING: could not read {p}, skipping", file=sys.stderr)
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


def _infer_static(
    client: object,
    images: list[npt.NDArray[np.uint8]],
) -> list[npt.NDArray[np.float32]]:
    """Run all frames through the static batch-8 model, return raw output0 rows."""
    from app.inference.detector import PersonDetector  # type: ignore[import]

    detector = PersonDetector(client, model_name=_STATIC_MODEL, static_batch_size=8, dynamic_batch=False)  # type: ignore[arg-type]

    import asyncio

    async def _run() -> list[list[object]]:
        return await detector.detect_batch(images)  # type: ignore[return-value]

    return asyncio.run(_run())  # type: ignore[return-value]


def _infer_dynamic(
    client: object,
    images: list[npt.NDArray[np.uint8]],
) -> list[list[object]]:
    from app.inference.detector import PersonDetector  # type: ignore[import]

    detector = PersonDetector(client, model_name=_DYNAMIC_MODEL, dynamic_batch=True)  # type: ignore[arg-type]

    import asyncio

    async def _run() -> list[list[object]]:
        return await detector.detect_batch(images)  # type: ignore[return-value]

    return asyncio.run(_run())


def _compare(
    static_results: list[list[object]],
    dynamic_results: list[list[object]],
) -> tuple[bool, str]:
    """Compare detection lists frame-by-frame. Returns (passed, report_table)."""
    rows = []
    rows.append(
        f"{'Frame':>6}  {'Static #':>8}  {'Dynamic #':>9}  "
        f"{'Unmatched S':>11}  {'Unmatched D':>11}  {'Conf MAE':>8}  {'Status':>8}"
    )
    rows.append("-" * 75)

    all_pass = True
    for i, (s_boxes, d_boxes) in enumerate(zip(static_results, dynamic_results, strict=True)):
        s = [(b.x1, b.y1, b.x2, b.y2, b.confidence) for b in s_boxes]  # type: ignore[attr-defined]
        d = [(b.x1, b.y1, b.x2, b.y2, b.confidence) for b in d_boxes]  # type: ignore[attr-defined]

        # Greedy IoU matching
        matched_s: set[int] = set()
        matched_d: set[int] = set()
        conf_deltas: list[float] = []
        for si, sb in enumerate(s):
            best_iou = 0.0
            best_di = -1
            for di, db in enumerate(d):
                if di in matched_d:
                    continue
                iou = _iou(np.array(sb[:4], dtype=np.float32), np.array(db[:4], dtype=np.float32))
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_iou >= _IOU_THRESHOLD and best_di >= 0:
                matched_s.add(si)
                matched_d.add(best_di)
                conf_deltas.append(abs(sb[4] - d[best_di][4]))

        unmatched_s = len(s) - len(matched_s)
        unmatched_d = len(d) - len(matched_d)
        conf_mae = float(np.mean(conf_deltas)) if conf_deltas else 0.0
        frame_pass = unmatched_s == 0 and unmatched_d == 0 and conf_mae <= _CONF_MAE_THRESHOLD
        if not frame_pass:
            all_pass = False
        status = "PASS" if frame_pass else "FAIL"
        rows.append(
            f"{i:>6}  {len(s):>8}  {len(d):>9}  "
            f"{unmatched_s:>11}  {unmatched_d:>11}  {conf_mae:>8.4f}  {status:>8}"
        )

    return all_pass, "\n".join(rows)


def main() -> None:
    triton_url = os.environ.get("TRITON_GRPC_URL", _DEFAULT_TRITON_URL)
    calib_dir_str = os.environ.get("DETECTOR_CALIB_IMAGES_DIR", "")
    n_frames = int(os.environ.get("DETECTOR_EQUIV_N_FRAMES", str(_DEFAULT_N_FRAMES)))

    if not calib_dir_str:
        print(
            "ERROR: set DETECTOR_CALIB_IMAGES_DIR to the directory of calibration images.",
            file=sys.stderr,
        )
        sys.exit(1)

    calib_dir = Path(calib_dir_str)
    if not calib_dir.is_dir():
        print(f"ERROR: {calib_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading up to {n_frames} frames from {calib_dir} ...")
    images = _load_images(calib_dir, n_frames)
    print(f"Loaded {len(images)} frames.")

    # Connect to Triton
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tracking-orchestrator"))
    try:
        from app.inference.triton_client import TritonGrpcClient  # type: ignore[import]
    except ImportError as exc:
        print(f"ERROR: could not import TritonGrpcClient: {exc}", file=sys.stderr)
        sys.exit(1)

    import asyncio

    async def _build_client() -> object:
        c = TritonGrpcClient(triton_url, timeout_ms=10000)
        await c.__aenter__()
        return c

    client = asyncio.run(_build_client())

    print(f"Running {len(images)} frames through {_STATIC_MODEL} (static) ...")
    static_results = _infer_static(client, images)
    print(f"Running {len(images)} frames through {_DYNAMIC_MODEL} (dynamic) ...")
    dynamic_results = _infer_dynamic(client, images)

    passed, table = _compare(static_results, dynamic_results)

    print()
    print(table)
    print()
    if passed:
        print(f"EQUIVALENCE GATE: PASS  ({len(images)} frames, IoU@{_IOU_THRESHOLD}, conf MAE <= {_CONF_MAE_THRESHOLD})")
        sys.exit(0)
    else:
        print(
            f"EQUIVALENCE GATE: FAIL  "
            f"(IoU@{_IOU_THRESHOLD}, conf MAE threshold {_CONF_MAE_THRESHOLD})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
