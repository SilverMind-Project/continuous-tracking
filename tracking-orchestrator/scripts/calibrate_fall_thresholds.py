"""Calibrate fall detection thresholds against labeled fixture sequences.

Runs FallFeatureExtractor + FallDetector over every *.jsonl in the sequences
directory (synthetic fixtures committed to git, optionally real privately recorded
sequences kept outside git) and prints a sensitivity / specificity table for a
small threshold grid.

Usage:

    cd tracking-orchestrator

    # Synthetic fixtures only (CI / development):
    uv run python scripts/calibrate_fall_thresholds.py

    # Include privately recorded sequences (outside git):
    uv run python scripts/calibrate_fall_thresholds.py \\
        --sequences-dir tests/fixtures/fall_sequences \\
        --extra-dir /private/real_fall_sequences

    # Print the table as CSV:
    uv run python scripts/calibrate_fall_thresholds.py --csv

Output: sensitivity / specificity table + recommended config block.

Expectation semantics (from fixture header):
    "detect"       TP when check_impact fires; FN otherwise.
    "no-detect"    TN when check_impact never fires; FP when it fires.
    "warning-max"  TN when is_escalatable never fires; FP when it fires.
                   (Warning without escalation is acceptable for this class.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.domain import BoundingBox  # noqa: E402
from app.inference.schemas import Keypoint  # noqa: E402
from app.trajectory.fall_detector import FallDetector, FallDetectorConfig  # noqa: E402
from app.trajectory.fall_features import FallFeatureExtractor, FallFrameInput  # noqa: E402
from app.trajectory.posture import PostureScores  # noqa: E402

_DEFAULT_SEQUENCES_DIR = _REPO_ROOT / "tests" / "fixtures" / "fall_sequences"
_RESTING_ROOMS = ("bed", "bedroom")


# ---------------------------------------------------------------------------
# Threshold grid
# ---------------------------------------------------------------------------

_DESCENT_RATES = [0.6, 0.7, 0.8, 0.9, 1.0]
_HEIGHT_RATIOS = [0.45, 0.50, 0.55, 0.60]
_LYING_SCORES = [0.3, 0.4, 0.5]

# Shipped defaults (centre of the grid).
_SHIPPED = FallDetectorConfig(
    max_descent_rate_hps_threshold=0.8,
    height_ratio_threshold=0.55,
    lying_score_threshold=0.4,
)


# ---------------------------------------------------------------------------
# Fixture loading (mirrors test_fall_sequences.py; kept separate to avoid
# a test-module import from a script)
# ---------------------------------------------------------------------------


def _deserialize_frame(d: dict) -> FallFrameInput:
    bd = d["bbox"]
    bbox = BoundingBox(
        x_min=int(bd["x_min"]),
        y_min=int(bd["y_min"]),
        x_max=int(bd["x_max"]),
        y_max=int(bd["y_max"]),
    )
    raw_kps = d["keypoints"]
    keypoints: tuple[Keypoint, ...] | None = None
    if raw_kps is not None:
        keypoints = tuple(
            Keypoint(x=float(k[0]), y=float(k[1]), score=float(k[2])) for k in raw_kps
        )
    raw_ps = d["posture_scores"]
    posture_scores: PostureScores | None = None
    if raw_ps is not None:
        posture_scores = PostureScores(
            lying=float(raw_ps["lying"]),
            sitting=float(raw_ps["sitting"]),
            standing_walking=float(raw_ps["standing_walking"]),
            keypoint_confidence=float(raw_ps.get("keypoint_confidence", 0.0)),
        )
    return FallFrameInput(
        captured_at=datetime.fromisoformat(d["captured_at"]),
        bbox=bbox,
        keypoints=keypoints,
        posture_scores=posture_scores,
        floor_speed_m_s=d["floor_speed_m_s"],
        motion_energy_nu_s=d["motion_energy_nu_s"],
    )


def _load_fixture(path: Path) -> tuple[dict, list[FallFrameInput]]:
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    frames = [_deserialize_frame(json.loads(ln)) for ln in lines[1:] if ln.strip()]
    return header, frames


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _FixtureResult:
    name: str
    expectation: str
    room: str
    any_detected: bool
    any_escalated: bool


def _eval_fixture(
    name: str,
    header: dict,
    frames: list[FallFrameInput],
    cfg: FallDetectorConfig,
) -> _FixtureResult:
    extractor = FallFeatureExtractor()
    detector = FallDetector(cfg)
    room = header.get("room", "living_room")
    any_det = False
    # confirmed_escalated: is_escalatable returned True AND post_event_motion_nu_s
    # was non-None at the same frame.  This mirrors the test semantics (posture-proxy
    # escalation before the motion window closes is not counted as an emergency FP).
    confirmed_esc = False
    for frame in frames:
        feat = extractor.update("ph-cal", frame)
        dec = detector.check_impact(feat, room, _RESTING_ROOMS)
        if dec is not None:
            any_det = True
            if feat.post_event_motion_nu_s is not None and detector.is_escalatable(feat):
                confirmed_esc = True
    return _FixtureResult(
        name=name,
        expectation=header["expectation"],
        room=room,
        any_detected=any_det,
        any_escalated=confirmed_esc,
    )


@dataclass
class _GridPoint:
    descent_rate: float
    height_ratio: float
    lying_score: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float

    def as_row(self) -> str:
        return (
            f"{self.descent_rate:.1f}  {self.height_ratio:.2f}  {self.lying_score:.1f}  "
            f"{self.sensitivity * 100:5.1f}%  {self.specificity * 100:5.1f}%  "
            f"TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn}"
        )

    def as_csv(self) -> str:
        return (
            f"{self.descent_rate},{self.height_ratio},{self.lying_score},"
            f"{self.sensitivity:.4f},{self.specificity:.4f},"
            f"{self.tp},{self.fp},{self.tn},{self.fn}"
        )


def _score(results: list[_FixtureResult]) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) from a list of evaluated fixture results."""
    tp = fp = tn = fn = 0
    for r in results:
        if r.expectation == "detect":
            if r.any_detected:
                tp += 1
            else:
                fn += 1
        elif r.expectation == "no-detect":
            if r.any_detected:
                fp += 1
            else:
                tn += 1
        elif r.expectation == "warning-max":
            # Emergency is the FP; warning without escalation is acceptable (TN).
            if r.any_escalated:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn


def _run_grid(
    fixtures: list[tuple[str, dict, list[FallFrameInput]]],
) -> list[_GridPoint]:
    points: list[_GridPoint] = []
    for dr, hr, ls in product(_DESCENT_RATES, _HEIGHT_RATIOS, _LYING_SCORES):
        cfg = FallDetectorConfig(
            max_descent_rate_hps_threshold=dr,
            height_ratio_threshold=hr,
            lying_score_threshold=ls,
        )
        results = [_eval_fixture(name, hdr, frames, cfg) for name, hdr, frames in fixtures]
        tp, fp, tn, fn = _score(results)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        points.append(_GridPoint(dr, hr, ls, tp, fp, tn, fn, sens, spec))
    return sorted(points, key=lambda p: (-(p.sensitivity + p.specificity), p.descent_rate))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=_DEFAULT_SEQUENCES_DIR,
        help="Directory of *.jsonl fixture files (default: tests/fixtures/fall_sequences/)",
    )
    parser.add_argument(
        "--extra-dir",
        type=Path,
        default=None,
        help="Optional extra directory of privately recorded sequences (not in git).",
    )
    parser.add_argument(
        "--csv", action="store_true", help="Print table as CSV instead of human-readable."
    )
    args = parser.parse_args(argv)

    dirs = [args.sequences_dir]
    if args.extra_dir:
        dirs.append(args.extra_dir)

    fixtures: list[tuple[str, dict, list[FallFrameInput]]] = []
    for d in dirs:
        if not d.exists():
            print(f"warning: sequences dir not found: {d}", file=sys.stderr)
            continue
        for path in sorted(d.glob("*.jsonl")):
            try:
                hdr, frames = _load_fixture(path)
                if "expectation" not in hdr:
                    print(f"skip {path.name}: no expectation header", file=sys.stderr)
                    continue
                fixtures.append((path.stem, hdr, frames))
            except (ValueError, KeyError, OSError) as exc:
                print(f"skip {path.name}: {exc}", file=sys.stderr)

    if not fixtures:
        print("no fixtures found — run synthesize_fall_sequence.py first", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(fixtures)} fixtures:")
    for name, hdr, frames in fixtures:
        exp = hdr["expectation"]
        room = hdr.get("room", "?")
        print(f"  {name:35s}  {exp:12s}  {len(frames):3d} frames  room={room}")

    n = len(_DESCENT_RATES) * len(_HEIGHT_RATIOS) * len(_LYING_SCORES)
    print(
        f"\nThreshold grid ({len(_DESCENT_RATES)}x{len(_HEIGHT_RATIOS)}x{len(_LYING_SCORES)}"
        f" = {n} combinations)\n"
    )

    points = _run_grid(fixtures)

    if args.csv:
        print("descent_rate,height_ratio,lying_score,sensitivity,specificity,TP,FP,TN,FN")
        for p in points:
            print(p.as_csv())
    else:
        print(f"{'descent':>8}  {'h_ratio':>7}  {'lying':>5}  {'sens':>6}  {'spec':>6}  counts")
        print("-" * 72)
        for p in points:
            marker = (
                " <-- shipped"
                if (p.descent_rate == 0.8 and p.height_ratio == 0.55 and p.lying_score == 0.4)
                else ""
            )
            print(p.as_row() + marker)

    # Shipped config evaluation.
    shipped_cfg = _SHIPPED
    shipped_results = [
        _eval_fixture(name, hdr, frames, shipped_cfg) for name, hdr, frames in fixtures
    ]
    tp, fp, tn, fn = _score(shipped_results)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n{'=' * 72}")
    print("SHIPPED THRESHOLDS  descent_rate=0.8  height_ratio=0.55  lying_score=0.4")
    print(f"  Sensitivity: {sens * 100:.1f}%  (TP={tp} / {tp + fn} positive fixtures)")
    print(f"  Specificity: {spec * 100:.1f}%  (TN={tn} / {tn + fp} negative+warning-max fixtures)")

    print("\nPer-fixture at shipped thresholds:")
    for r in shipped_results:
        status = (
            "  OK "
            if (
                (r.expectation == "detect" and r.any_detected)
                or (r.expectation == "no-detect" and not r.any_detected)
                or (r.expectation == "warning-max" and not r.any_escalated)
            )
            else "FAIL"
        )
        print(
            f"  [{status}]  {r.name:35s}  {r.expectation:12s}  "
            f"detected={r.any_detected}  escalated={r.any_escalated}"
        )

    print("\nRecommended config block (best sensitivity+specificity from grid):")
    best = points[0]
    print(f"""
fall_detection:
  enabled: false   # Enable per-deployment after fixture proofs pass (see runbook below).
  max_descent_rate_hps_threshold: {best.descent_rate}
  height_ratio_threshold: {best.height_ratio}
  lying_score_threshold: {best.lying_score}
  # sensitivity={best.sensitivity * 100:.1f}%  specificity={best.specificity * 100:.1f}%
  # TP={best.tp}  FP={best.fp}  TN={best.tn}  FN={best.fn}
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
