"""Synthesize parametric fall-sequence fixtures for fall detector tests.

Format: JSON-lines.  First line = sidecar header with expectation metadata;
remaining lines = serialised FallFrameInput frames.

Header line:
    {"type": "header", "expectation": "detect" | "no-detect" | "warning-max",
     "description": "...", "room": "..."}

Frame line:
    {"captured_at": "<ISO-8601+TZ>",
     "bbox": {"x_min": int, "y_min": int, "x_max": int, "y_max": int},
     "keypoints": [[x, y, score], ...]  # 17 COCO kps in crop coords, or null,
     "posture_scores": {"lying": f, "sitting": f, "standing_walking": f,
                        "keypoint_confidence": f} | null,
     "floor_speed_m_s": float | null,
     "motion_energy_nu_s": float | null}

Expectation semantics (enforced by tests/integration/test_fall_sequences.py):
    "detect"      - check_impact returns non-None for at least one frame.
    "no-detect"   - check_impact returns None for every frame.
    "warning-max" - check_impact may fire (warning) but is_escalatable never
                    returns True when the person is still low (no emergency paging).

Regenerate:

    cd tracking-orchestrator
    uv run python scripts/synthesize_fall_sequence.py

Committed to git; regenerate when FallFrameInput schema changes or new scenarios
are needed for calibration.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "fall_sequences"

_BASE_TIME = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
_X_MIN = 100
_BBOX_WIDTH = 120
_NUM_KPS = 17

# Canonical standing geometry (mirrors test_fall_features.py conventions so the
# extractor's h_est math can be checked against those unit tests).
_H_STAND = 300.0  # bbox height when upright, px
_H_LYING = 120.0  # bbox height when flat on floor, px
_HIP_Y_STAND = 250.0  # absolute image-y of the hip proxy while standing (down=positive)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _kps(crop_y: float = 0.5, score: float = 0.9) -> list[list[float]]:
    """17 COCO keypoints all at crop-y (all body groups at the same level).

    Placing all keypoints at crop-y 0.5 makes vertical_y == bbox centre-y ==
    hip_y_abs, which reproduces the convention in test_fall_features.py and
    keeps height_above_floor_px = bbox_height / 2.
    """
    return [[0.5, crop_y, score]] * _NUM_KPS


def _bbox(hip_y_abs: float, height: float) -> dict[str, int]:
    """Axis-aligned bbox with hip at crop-y 0.5 and fixed x extent."""
    y_min = round(hip_y_abs - height / 2.0)
    y_max = round(hip_y_abs + height / 2.0)
    return {"x_min": _X_MIN, "y_min": y_min, "x_max": _X_MIN + _BBOX_WIDTH, "y_max": y_max}


def _posture(
    lying: float = 0.0,
    sitting: float = 0.0,
    standing: float | None = None,
    kp_conf: float = 0.8,
) -> dict[str, float]:
    sw = standing if standing is not None else max(0.0, round(1.0 - lying - sitting, 4))
    return {
        "lying": round(lying, 4),
        "sitting": round(sitting, 4),
        "standing_walking": round(sw, 4),
        "keypoint_confidence": kp_conf,
    }


def _frame(
    t_s: float,
    *,
    hip_y: float,
    height: float,
    lying: float = 0.0,
    sitting: float = 0.0,
    keypoints: bool = True,
    posture_none: bool = False,
    floor_speed: float | None = 0.5,
    motion: float | None = 0.01,
) -> dict:
    """Serialise one FallFrameInput-equivalent as a plain dict."""
    captured_at = (_BASE_TIME + timedelta(seconds=t_s)).isoformat()
    return {
        "captured_at": captured_at,
        "bbox": _bbox(hip_y, height),
        "keypoints": _kps() if keypoints else None,
        "posture_scores": None if posture_none else _posture(lying=lying, sitting=sitting),
        "floor_speed_m_s": floor_speed,
        "motion_energy_nu_s": motion,
    }


def _header(expectation: str, description: str, room: str) -> dict:
    return {
        "type": "header",
        "expectation": expectation,
        "description": description,
        "room": room,
    }


def _write(path: Path, header: dict, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for fr in frames:
            fh.write(json.dumps(fr) + "\n")
    print(f"wrote {path.name}: {len(frames)} frames")


# ---------------------------------------------------------------------------
# Positive fixtures - must produce check_impact != None on at least one frame
# ---------------------------------------------------------------------------


def fall_forward_fast() -> tuple[dict, list[dict]]:
    """Fast frontal fall: 0.4 s descent, ~1.5 hps peak adjacent-pair rate."""
    header = _header("detect", "fast forward fall, ~1.5 hps peak rate", "living_room")
    frames: list[dict] = []
    dt = 0.1
    drop_px = 0.6 * _H_STAND  # 180 px; at 10 fps per-pair rate = 45/300/0.1 = 1.5 hps

    t_stand, t_drop, t_post = 1.5, 0.4, 1.0
    total = t_stand + t_drop + t_post
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
        elif t < t_stand + t_drop:
            frac = (t - t_stand) / t_drop
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (_H_LYING - _H_STAND)
            lying = frac * 0.8
        else:
            hip, h, lying = _HIP_Y_STAND + drop_px, _H_LYING, 0.8
        frames.append(_frame(t, hip_y=hip, height=h, lying=lying, floor_speed=0.5, motion=0.01))
        t = round(t + dt, 9)
    return header, frames


def fall_slump_slow() -> tuple[dict, list[dict]]:
    """Slow two-phase slump: 1.2 s total descent; peak rate ~0.92 hps.

    Phase A (0.6 s) - gentle forward lean (0.15 h).
    Phase B (0.6 s) - rapid terminal collapse (0.55 h) => ~0.92 hps peak.

    Post-event motion is above the stillness floor (0.12 nu/s > 0.05).  The
    post_event_motion_nu_s window (2.0 s) closes during the sequence, so at
    the frames where check_impact fires WITH a known post_event_motion, the
    value is 0.095 nu/s > 0.05 => is_escalatable returns False.

    t_post is set to 2.0 s so the sequence covers event_ts + 2.0 s comfortably.
    """
    header = _header(
        "warning-max",
        "slow two-phase slump 1.2 s, detects at warning; post-motion blocks confirmed escalation",
        "living_room",
    )
    frames: list[dict] = []
    dt = 0.1

    t_stand, t_lean, t_drop, t_post = 1.0, 0.6, 0.6, 2.0
    lean_px = 0.15 * _H_STAND  # 45 px - slow phase
    drop_px = 0.55 * _H_STAND  # 165 px - fast phase; 165/(6 frames) = 27.5/frame = 0.917 hps

    total = t_stand + t_lean + t_drop + t_post
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
            post = False
        elif t < t_stand + t_lean:
            frac = (t - t_stand) / t_lean
            hip = _HIP_Y_STAND + frac * lean_px
            h = _H_STAND - frac * 20.0
            lying = frac * 0.1
            post = False
        elif t < t_stand + t_lean + t_drop:
            frac = (t - t_stand - t_lean) / t_drop
            hip = _HIP_Y_STAND + lean_px + frac * drop_px
            h = (_H_STAND - 20.0) + frac * (_H_LYING - (_H_STAND - 20.0))
            lying = 0.1 + frac * 0.65
            post = False
        else:
            hip = _HIP_Y_STAND + lean_px + drop_px
            h = _H_LYING
            lying = 0.75
            post = True
        # Post-event motion > stillness_motion_floor keeps is_escalatable False.
        frames.append(
            _frame(
                t, hip_y=hip, height=h, lying=lying, floor_speed=0.4, motion=0.12 if post else 0.02
            )
        )
        t = round(t + dt, 9)
    return header, frames


def fall_with_pose_loss() -> tuple[dict, list[dict]]:
    """Standard fast fall where keypoints disappear one frame after impact.

    The first post-impact frame retains keypoints to anchor the low height_ratio
    (~0.4), then pose is lost (person flat on floor, RTMPose drops keypoints).
    Rule 4 passes via the pose-unavailable branch for subsequent frames.
    """
    header = _header(
        "detect",
        "fast fall with pose loss at impact; rule 4 passes via no-pose branch",
        "living_room",
    )
    frames: list[dict] = []
    dt = 0.1
    drop_px = 0.6 * _H_STAND
    t_stand, t_drop = 1.5, 0.4
    # Lose keypoints one frame after first post-impact frame.
    t_pose_loss = t_stand + t_drop + dt + dt

    total = t_stand + t_drop + 1.5
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
        elif t < t_stand + t_drop:
            frac = (t - t_stand) / t_drop
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (_H_LYING - _H_STAND)
            lying = frac * 0.7
        else:
            hip, h, lying = _HIP_Y_STAND + drop_px, _H_LYING, 0.8

        has_kps = t < t_pose_loss
        frames.append(
            _frame(
                t,
                hip_y=hip,
                height=h,
                lying=lying if has_kps else 0.0,
                keypoints=has_kps,
                posture_none=not has_kps,
                floor_speed=0.5,
                motion=0.01,
            )
        )
        t = round(t + dt, 9)
    return header, frames


def fall_low_fps() -> tuple[dict, list[dict]]:
    """Fall captured at 2 fps (motion-gated or low-cadence camera).

    Adjacent pair (t=2.0 -> t=2.5): 180 px hip drop in 0.5 s => 1.2 hps.
    Five standing frames ensure min_samples (5) is satisfied at the impact frame.
    """
    drop_px = 0.6 * _H_STAND

    timeline = [
        (0.0, _HIP_Y_STAND, _H_STAND, 0.0),
        (0.5, _HIP_Y_STAND, _H_STAND, 0.0),
        (1.0, _HIP_Y_STAND, _H_STAND, 0.0),
        (1.5, _HIP_Y_STAND, _H_STAND, 0.0),
        (2.0, _HIP_Y_STAND, _H_STAND, 0.0),
        (2.5, _HIP_Y_STAND + drop_px, _H_LYING, 0.8),  # impact frame
        (3.0, _HIP_Y_STAND + drop_px, _H_LYING, 0.8),
        (3.5, _HIP_Y_STAND + drop_px, _H_LYING, 0.8),
    ]
    frames = [
        _frame(t, hip_y=hip, height=h, lying=lying, floor_speed=0.5, motion=0.01)
        for t, hip, h, lying in timeline
    ]
    return _header("detect", "fall at 2 fps, adjacent-pair rate ~1.2 hps", "living_room"), frames


# ---------------------------------------------------------------------------
# Guardrail fixtures - must NOT trigger (or only trigger warning, never emergency)
# ---------------------------------------------------------------------------


def sit_down_normal() -> tuple[dict, list[dict]]:
    """Slow controlled sit: descent ~0.13 hps, well below the 0.8 threshold."""
    header = _header("no-detect", "normal slow sit-down, descent ~0.13 hps", "living_room")
    frames: list[dict] = []
    dt = 0.1

    t_stand, t_sit, t_hold = 1.0, 2.0, 1.5
    drop_px, h_sit = 40.0, 220.0  # hip drops 40 px over 2 s = 0.13 hps; height 300 -> 220

    total = t_stand + t_sit + t_hold
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying, sitting = _HIP_Y_STAND, _H_STAND, 0.0, 0.0
        elif t < t_stand + t_sit:
            frac = (t - t_stand) / t_sit
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (h_sit - _H_STAND)
            lying, sitting = 0.0, frac * 0.7
        else:
            hip, h, lying, sitting = _HIP_Y_STAND + drop_px, h_sit, 0.0, 0.7
        frames.append(
            _frame(
                t, hip_y=hip, height=h, lying=lying, sitting=sitting, floor_speed=0.3, motion=0.1
            )
        )
        t = round(t + dt, 9)
    return header, frames


def sit_down_heavy() -> tuple[dict, list[dict]]:
    """Heavy fast plop into an armchair.

    Descent rate reaches ~2.0 hps (rule 2 fires), but:
    - height_above_floor stays at 100 px -> ratio 100/150 = 0.67 > 0.55 (rule 3 vetoes).
    - lying_score stays at 0.05 (seated posture, rule 4 vetoes).
    Both rule 3 and rule 4 independently prevent check_impact from returning.
    """
    header = _header(
        "no-detect",
        "heavy fast plop: height_ratio 0.67 and lying_score 0.05 veto detection",
        "living_room",
    )
    frames: list[dict] = []
    dt = 0.1

    # Hip drops 60 px in 0.2 s => 60/300/0.1 = 2.0 hps, but stays seated after.
    t_stand, t_plop, t_hold = 1.5, 0.2, 1.5
    drop_px, h_sit = 60.0, 200.0  # height_above_floor = 100 px => ratio 0.67 > 0.55

    total = t_stand + t_plop + t_hold
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying, sitting = _HIP_Y_STAND, _H_STAND, 0.0, 0.0
        elif t < t_stand + t_plop:
            frac = (t - t_stand) / t_plop
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (h_sit - _H_STAND)
            lying, sitting = 0.02, frac * 0.6
        else:
            hip, h, lying, sitting = _HIP_Y_STAND + drop_px, h_sit, 0.05, 0.65
        # Post-plop: person adjusts in chair -> high motion, not escalatable.
        frames.append(
            _frame(
                t, hip_y=hip, height=h, lying=lying, sitting=sitting, floor_speed=0.3, motion=0.2
            )
        )
        t = round(t + dt, 9)
    return header, frames


def lie_on_bed() -> tuple[dict, list[dict]]:
    """Person lies down on bed: physically identical to a fall but room="bedroom".

    Rule 6 (resting-room veto) blocks detection regardless of other features.
    """
    header = _header(
        "no-detect", "lying down on bed; resting-room veto blocks detection", "bedroom"
    )
    frames: list[dict] = []
    dt = 0.1
    drop_px = 0.6 * _H_STAND
    t_stand, t_drop, t_post = 1.5, 0.4, 1.0

    total = t_stand + t_drop + t_post
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
        elif t < t_stand + t_drop:
            frac = (t - t_stand) / t_drop
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (_H_LYING - _H_STAND)
            lying = frac * 0.8
        else:
            hip, h, lying = _HIP_Y_STAND + drop_px, _H_LYING, 0.8
        frames.append(_frame(t, hip_y=hip, height=h, lying=lying, floor_speed=0.3, motion=0.01))
        t = round(t + dt, 9)
    return header, frames


def bend_to_pick_up() -> tuple[dict, list[dict]]:
    """Bend down to pick up an object then return to standing.

    Descent rate ~0.125 hps (controlled, well below 0.8); height recovers fully.
    """
    header = _header(
        "no-detect",
        "bend and return: rate ~0.125 hps, height fully recovers",
        "kitchen",
    )
    frames: list[dict] = []
    dt = 0.1

    t_stand, t_bend, t_hold, t_return, t_post = 1.0, 0.8, 0.3, 0.8, 1.0
    drop_px = 0.3 * _H_STAND  # 90 px; over 0.8 s at 10 fps = 11.25 px/frame = 0.375 hps

    total = t_stand + t_bend + t_hold + t_return + t_post
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
        elif t < t_stand + t_bend:
            frac = (t - t_stand) / t_bend
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND - frac * 30.0
            lying = frac * 0.15
        elif t < t_stand + t_bend + t_hold:
            hip, h, lying = _HIP_Y_STAND + drop_px, _H_STAND - 30.0, 0.15
        elif t < t_stand + t_bend + t_hold + t_return:
            frac = (t - t_stand - t_bend - t_hold) / t_return
            hip = _HIP_Y_STAND + drop_px * (1.0 - frac)
            h = (_H_STAND - 30.0) + frac * 30.0
            lying = 0.15 * (1.0 - frac)
        else:
            hip, h, lying = _HIP_Y_STAND, _H_STAND, 0.0
        frames.append(_frame(t, hip_y=hip, height=h, lying=lying, floor_speed=0.2, motion=0.15))
        t = round(t + dt, 9)
    return header, frames


def tie_shoes() -> tuple[dict, list[dict]]:
    """Kneel to tie shoes: height drops but descent rate ~0.4 hps and lying_score ~0.1.

    Even if height_ratio briefly dips near 0.55, lying_score (rule 4) vetoes detection.
    """
    header = _header(
        "no-detect",
        "kneel to tie shoes: rate ~0.4 hps and low lying_score veto",
        "hallway",
    )
    frames: list[dict] = []
    dt = 0.1

    t_stand, t_kneel, t_hold, t_rise, t_post = 1.0, 1.0, 2.0, 1.0, 0.5
    drop_px = 0.4 * _H_STAND  # 120 px; over 1.0 s = 12 px/frame = 0.4 hps
    h_kneeling = 180.0

    total = t_stand + t_kneel + t_hold + t_rise + t_post
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip, h, lying, sitting = _HIP_Y_STAND, _H_STAND, 0.0, 0.0
        elif t < t_stand + t_kneel:
            frac = (t - t_stand) / t_kneel
            hip = _HIP_Y_STAND + frac * drop_px
            h = _H_STAND + frac * (h_kneeling - _H_STAND)
            lying, sitting = frac * 0.1, frac * 0.4
        elif t < t_stand + t_kneel + t_hold:
            hip, h, lying, sitting = _HIP_Y_STAND + drop_px, h_kneeling, 0.1, 0.4
        elif t < t_stand + t_kneel + t_hold + t_rise:
            frac = (t - t_stand - t_kneel - t_hold) / t_rise
            hip = _HIP_Y_STAND + drop_px * (1.0 - frac)
            h = h_kneeling + frac * (_H_STAND - h_kneeling)
            lying, sitting = 0.1 * (1.0 - frac), 0.4 * (1.0 - frac)
        else:
            hip, h, lying, sitting = _HIP_Y_STAND, _H_STAND, 0.0, 0.0
        frames.append(
            _frame(
                t, hip_y=hip, height=h, lying=lying, sitting=sitting, floor_speed=0.1, motion=0.2
            )
        )
        t = round(t + dt, 9)
    return header, frames


def child_or_pet_proxy() -> tuple[dict, list[dict]]:
    """Small fast-moving entity (child or pet): small bbox, erratic motion.

    Must not crash the extractor and must not trigger a fall alert.
    h_est anchors to the entity's own small scale so height_ratio stays near 1.0.
    """
    header = _header(
        "no-detect",
        "child/pet proxy: small bbox, erratic motion, height_ratio near 1.0",
        "living_room",
    )
    frames: list[dict] = []
    dt = 0.1
    rng = np.random.default_rng(42)

    base_hip, base_h = 150.0, 100.0  # small entity
    total = 3.0
    t = 0.0
    while t <= total + 1e-9:
        hip = float(base_hip + rng.normal(0, 15.0))
        h = float(max(base_h + rng.normal(0, 5.0), 40.0))
        frames.append(
            _frame(
                t,
                hip_y=hip,
                height=h,
                lying=0.0,
                sitting=0.0,
                floor_speed=float(abs(rng.normal(0.2, 0.1))),
                motion=float(abs(rng.normal(0.3, 0.1))),
            )
        )
        t = round(t + dt, 9)
    return header, frames


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (fall_forward_fast, "fall_forward_fast.jsonl"),
        (fall_slump_slow, "fall_slump_slow.jsonl"),
        (fall_with_pose_loss, "fall_with_pose_loss.jsonl"),
        (fall_low_fps, "fall_low_fps.jsonl"),
        (sit_down_normal, "sit_down_normal.jsonl"),
        (sit_down_heavy, "sit_down_heavy.jsonl"),
        (lie_on_bed, "lie_on_bed.jsonl"),
        (bend_to_pick_up, "bend_to_pick_up.jsonl"),
        (tie_shoes, "tie_shoes.jsonl"),
        (child_or_pet_proxy, "child_or_pet_proxy.jsonl"),
    ]

    for fn, name in scenarios:
        hdr, frames = fn()
        _write(_FIXTURES_DIR / name, hdr, frames)

    print(f"\ndone — {len(scenarios)} fixtures in {_FIXTURES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
