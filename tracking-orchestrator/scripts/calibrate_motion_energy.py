"""Offline calibration script for motion-energy unit thresholds.

Reads labeled (pose, bbox, timestamp) sequences and reports the distribution
of ``mean_keypoint_velocity_nu_s`` for each behavior segment, then recommends
``_STILL_VELOCITY_FLOOR_NU_S`` and ``_WALKING_VELOCITY_NU_S`` values.

Usage (analytical mode -- no live data required)::

    uv run python -m scripts.calibrate_motion_energy --analytic

Usage (fixture mode)::

    uv run python -m scripts.calibrate_motion_energy \\
        --fixture tests/fixtures/frame_replays/single_camera_turn.bin \\
        --label-csv /path/to/labels.csv

Label CSV schema (no header, comma-separated)::

    start_unix_s,end_unix_s,segment_label
    1748000000.0,1748000030.0,still
    1748000030.0,1748000060.0,walking

The script prints a distribution table and writes recommended values to stdout.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Ensure the tracking-orchestrator package root is on sys.path when run as a script.
_SCRIPT_DIR = Path(__file__).parent
_PKG_ROOT = _SCRIPT_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


class _Segment(NamedTuple):
    label: str  # "still" or "walking"
    velocities: list[float]


def _analytic_calibration() -> None:
    """Derive provisional thresholds analytically and print the result table.

    Assumptions (typical household CCTV, 5 fps, 1080p):
      - 1.7 m adult; bbox ~150x300 px → diag ~335 px.
      - Still: keypoints jitter ~2-3 px absolute per frame due to breathing.
      - Walking 1 m/s: body translation ~35 px/frame (0.2 m * 300/1.7)
        plus ~10 px/frame limb swing; 12-joint mean ~40 px/frame.

    Formula: velocity_nu_s = mean_displacement_px / bbox_diag_px * fps
    """
    fps = 5.0
    bbox_diag = 335.0  # sqrt(150^2 + 300^2)

    still_samples = [1.5, 2.0, 2.5, 3.0, 3.5]  # px/frame, representative still range
    walking_samples = [30.0, 35.0, 40.0, 45.0, 50.0]  # px/frame, representative walking range

    still_vels = [s / bbox_diag * fps for s in still_samples]
    walking_vels = [s / bbox_diag * fps for s in walking_samples]

    _print_distribution("still (analytic)", still_vels)
    _print_distribution("walking (analytic)", walking_vels)

    still_p95 = float(np.percentile(still_vels, 95))
    walking_p5 = float(np.percentile(walking_vels, 5))
    recommended_floor = _round_up(still_p95, 0.005)
    recommended_walking = _round_up((still_p95 + walking_p5) / 2.0, 0.005)

    print()
    print("Recommended (provisional -- recalibrate after 1 week of live data):")
    print(f"  _STILL_VELOCITY_FLOOR_NU_S = {recommended_floor:.3f}  # p95 of still segments")
    print(
        f"  _WALKING_VELOCITY_NU_S     = {recommended_walking:.3f}"
        "  # midpoint still-p95 / walking-p5"
    )
    print()
    print("Arithmetic:")
    print(f"  bbox_diag ≈ {bbox_diag:.0f} px  (150x300 px bbox, typical household camera)")
    print(
        f"  still: {still_samples[2]:.1f} px/frame / {bbox_diag:.0f} * {fps:.0f} fps"
        f" = {still_samples[2] / bbox_diag * fps:.4f} nu/s (median)"
    )
    print(
        f"  walking: {walking_samples[2]:.1f} px/frame / {bbox_diag:.0f} * {fps:.0f} fps"
        f" = {walking_samples[2] / bbox_diag * fps:.4f} nu/s (median)"
    )


def _print_distribution(label: str, values: list[float]) -> None:
    arr = np.array(values)
    print(
        f"  {label:30s}  "
        f"p5={float(np.percentile(arr, 5)):.4f}  "
        f"p50={float(np.percentile(arr, 50)):.4f}  "
        f"p95={float(np.percentile(arr, 95)):.4f}  "
        f"n={len(values)}"
    )


def _round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _load_fixture_segments(fixture_path: Path, label_csv: Path | None) -> list[_Segment]:
    """Load (pose, bbox, timestamp) from a frame-replay fixture and split into labeled segments."""
    # Import here so the analytic path works without triton_shared installed.
    from app.domain import BoundingBox
    from app.inference.schemas import Keypoint, PoseResult
    from app.trajectory.motion_energy import MotionEnergyTracker

    # Read label windows.
    labels: list[tuple[float, float, str]] = []
    if label_csv is not None:
        for line in label_csv.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) == 3:
                start, end, seg_label = float(parts[0]), float(parts[1]), parts[2].strip()
                labels.append((start, end, seg_label))

    if not labels:
        print(
            f"[warn] No label CSV provided or CSV is empty. "
            f"Will report overall distribution for {fixture_path.name}.",
            file=sys.stderr,
        )
        labels = [(0.0, 1e18, "unlabeled")]

    # Parse the fixture: length-prefixed protobuf frames.
    from app.proto.frame_pb2 import FrameReady  # type: ignore[import]

    raw = fixture_path.read_bytes()
    pos = 0
    tracker = MotionEnergyTracker()

    segment_vels: dict[str, list[float]] = {seg_label: [] for _, _, seg_label in labels}

    while pos + 4 <= len(raw):
        (msg_len,) = struct.unpack_from(">I", raw, pos)
        pos += 4
        if pos + msg_len > len(raw):
            break
        msg_bytes = raw[pos : pos + msg_len]
        pos += msg_len

        frame = FrameReady()
        frame.ParseFromString(msg_bytes)
        ts = datetime.fromtimestamp(frame.captured_at_ns / 1e9, tz=UTC)
        ts_unix = ts.timestamp()

        # Determine segment label for this frame's timestamp.
        seg_label = next(
            (sl for s, e, sl in labels if s <= ts_unix < e),
            None,
        )
        if seg_label is None:
            continue

        # Use the first detected person's pose + bbox (if available).
        if not frame.detections:
            continue
        det = frame.detections[0]
        if not det.keypoints:
            continue

        bbox = BoundingBox(
            x_min=int(det.bbox.x_min),
            y_min=int(det.bbox.y_min),
            x_max=int(det.bbox.x_max),
            y_max=int(det.bbox.y_max),
        )
        kps = tuple(
            Keypoint(x=float(k.x), y=float(k.y), score=float(k.score)) for k in det.keypoints[:17]
        )
        if len(kps) < 17:
            continue
        pose = PoseResult(keypoints=kps)
        gt_id = det.global_track_id or "unknown"
        energy = tracker.update(gt_id, pose, ts, bbox)
        if energy.sample_count >= 2:
            segment_vels[seg_label].append(energy.mean_keypoint_velocity_nu_s)

    return [_Segment(label=sl, velocities=vels) for sl, vels in segment_vels.items() if vels]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--analytic",
        action="store_true",
        help="Run analytical calibration only (no fixture)",
    )
    parser.add_argument("--fixture", type=Path, help="Path to frame_replay .bin fixture")
    parser.add_argument(
        "--label-csv",
        type=Path,
        dest="label_csv",
        help="Path to label CSV",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Motion-energy calibration")
    print("Unit: mean keypoint displacement in normalized units per second (nu/s)")
    print("=" * 70)
    print()

    if args.analytic or args.fixture is None:
        _analytic_calibration()

    if args.fixture is not None:
        print(f"Fixture: {args.fixture.name}")
        segments = _load_fixture_segments(args.fixture, args.label_csv)
        if not segments:
            print("  [warn] No labeled frames found.", file=sys.stderr)
            return
        print("Distribution by segment label:")
        for seg in segments:
            _print_distribution(seg.label, seg.velocities)
        # Recommend from fixture data if still/walking labels present.
        still_seg = next((s for s in segments if "still" in s.label), None)
        walk_seg = next((s for s in segments if "walk" in s.label), None)
        if still_seg and walk_seg:
            still_p95 = float(np.percentile(still_seg.velocities, 95))
            walk_p5 = float(np.percentile(walk_seg.velocities, 5))
            rec_floor = _round_up(still_p95, 0.005)
            rec_walk = _round_up((still_p95 + walk_p5) / 2.0, 0.005)
            print()
            print("Recommended (from fixture data):")
            print(f"  _STILL_VELOCITY_FLOOR_NU_S = {rec_floor:.3f}")
            print(f"  _WALKING_VELOCITY_NU_S     = {rec_walk:.3f}")
        print()
        print("Provenance:")
        print(f"  fixture: {args.fixture.resolve()}")
        if args.label_csv:
            print(f"  labels:  {args.label_csv.resolve()}")


if __name__ == "__main__":
    _main()
