"""Synthesize deterministic WorldObservation replay fixtures for integration tests.

Format: length-prefixed JSON binary. Each chunk is one "frame step" (one call
to WorldTracker.step()), encoded as a JSON array of observation dicts:

    [u32 BE length][JSON bytes] [u32 BE length][JSON bytes] ...

Each observation dict contains:
    camera_id               str
    frame_index             int
    captured_at_iso         str   (ISO-8601 with timezone)
    floor_x_mm              int
    floor_y_mm              int
    embedding               list[float]
    detection_confidence    float
    bbox                    dict  x_min/y_min/y_max/y_max
    detection_id            str   (deterministic; used by truth sidecar)
    quality                 float
    calibrated              bool  (default True)
    floor_cov_random        list[float] | None  row-major 2x2 covariance in m²
    footpoint_reliable      bool
    face_anchor             dict | None
    orientation             int   (OrientationBin value; 4=UNKNOWN default)
    orientation_confidence  float (default 0.0)

Each fixture also gets a sidecar ``<name>.truth.json``:
    persons:          list[str]  true person labels
    detection_truth:  dict[str, str]  detection_id → person label
    events:           list[dict]  handoffs/exits

Run this script directly to regenerate the fixtures:

    cd tracking-orchestrator
    uv run python scripts/synthesize_replay_fixture.py

Fixtures are committed to git. Regenerate when WorldObservation schema changes.
"""

from __future__ import annotations

import json
import struct
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "frame_replays"

BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
FRAME_INTERVAL_S = 0.5

# OrientationBin values (mirrors app.domain.OrientationBin)
ORI_FRONT = 0
ORI_BACK = 1
ORI_LEFT = 2
ORI_RIGHT = 3
ORI_UNKNOWN = 4


def _obs(
    camera_id: str,
    frame_index: int,
    step: int,
    floor_x_mm: int,
    floor_y_mm: int,
    embedding: list[float],
    *,
    calibrated: bool = True,
    face_anchor: dict | None = None,
    detection_id: str,
    detection_confidence: float = 0.92,
    quality: float = 0.5,
    floor_cov_random: list[float] | None = None,
    footpoint_reliable: bool | None = None,
    orientation: int = ORI_UNKNOWN,
    orientation_confidence: float = 0.0,
    floor_residual_m: float | None = None,
) -> dict:
    obs_time = BASE_TIME + timedelta(seconds=step * FRAME_INTERVAL_S)
    if footpoint_reliable is None:
        footpoint_reliable = calibrated
    if floor_cov_random is None and calibrated:
        floor_cov_random = [0.04, 0.0, 0.0, 0.04]
    result: dict = {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "captured_at_iso": obs_time.isoformat(),
        "floor_x_mm": floor_x_mm,
        "floor_y_mm": floor_y_mm,
        "embedding": embedding,
        "detection_confidence": detection_confidence,
        "bbox": {"x_min": 100, "y_min": 100, "x_max": 300, "y_max": 400},
        "detection_id": detection_id,
        "quality": quality,
        "calibrated": calibrated,
        "floor_cov_random": floor_cov_random,
        "footpoint_reliable": footpoint_reliable,
        "orientation": orientation,
        "orientation_confidence": orientation_confidence,
    }
    if floor_residual_m is not None:
        result["floor_residual_m"] = floor_residual_m
    if face_anchor is not None:
        fa = dict(face_anchor)
        if "captured_at_iso" not in fa:
            fa["captured_at_iso"] = obs_time.isoformat()
        result["face_anchor"] = fa
    return result


def _write(path: Path, frames: list[list[dict]], truth: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for observations in frames:
            data = json.dumps(observations).encode()
            f.write(struct.pack(">I", len(data)))
            f.write(data)
    size_kb = path.stat().st_size // 1024
    print(f"wrote {path.name}: {len(frames)} steps, {size_kb} KB")

    truth_path = path.with_suffix(".truth.json")
    with truth_path.open("w") as tf:
        json.dump(truth, tf, indent=2)
    print(f"wrote {truth_path.name}")


def _make_truth(persons: list[str], frames: list[list[dict]], labels: list[list[str]]) -> dict:
    """Build the truth sidecar from per-frame per-observation person labels."""
    detection_truth: dict[str, str] = {}
    for frame_obs, frame_labels in zip(frames, labels, strict=True):
        for obs, label in zip(frame_obs, frame_labels, strict=True):
            detection_truth[obs["detection_id"]] = label
    return {"persons": persons, "detection_truth": detection_truth, "events": []}


def _make_truth_with_events(
    persons: list[str],
    frames: list[list[dict]],
    labels: list[list[str]],
    events: list[dict],
) -> dict:
    t = _make_truth(persons, frames, labels)
    t["events"] = events
    return t


# ── Existing fixtures (unchanged behavior) ──────────────────────────────


def two_cameras_one_room() -> tuple[list[list[dict]], dict]:
    """One person walking under two overlapping cameras in one room."""
    emb_a = [0.90, 0.10, 0.00]
    emb_a2 = [0.88, 0.12, 0.00]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(10):
        fx = 3000 + i * 200
        det_id = f"det-2c1r-c1-{i}"
        frames.append([_obs("cam-1", i, i, fx, 5000, emb_a, detection_id=det_id)])
        labels.append(["p1"])

    for i in range(10):
        step = 10 + i
        fx = 5000 + i * 200
        det_id_1 = f"det-2c1r-c1-{step}"
        det_id_2 = f"det-2c1r-c2-{step}"
        frames.append(
            [
                _obs("cam-1", step, step, fx, 5000, emb_a, detection_id=det_id_1),
                _obs("cam-2", step, step, fx, 5000, emb_a2, detection_id=det_id_2),
            ]
        )
        labels.append(["p1", "p1"])

    for i in range(10):
        step = 20 + i
        fx = 7000 + i * 200
        det_id = f"det-2c1r-c2-{step}"
        frames.append([_obs("cam-2", step, step, fx, 5000, emb_a2, detection_id=det_id)])
        labels.append(["p1"])

    truth = _make_truth(["p1"], frames, labels)
    return frames, truth


def two_rooms_two_people() -> tuple[list[list[dict]], dict]:
    """Two people in two non-overlapping rooms that never merge."""
    emb_a = [1.00, 0.00, 0.00]
    emb_b = [0.00, 1.00, 0.00]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(20):
        step = i
        fxa = 2000 + i * 50
        fxb = 15000 + i * 50
        det_a = f"det-2r2p-a-{i}"
        det_b = f"det-2r2p-b-{i}"
        frames.append(
            [
                _obs("cam-1", step, step, fxa, 2000, emb_a, detection_id=det_a),
                _obs("cam-2", step, step, fxb, 15000, emb_b, detection_id=det_b),
            ]
        )
        labels.append(["p1", "p2"])

    truth = _make_truth(["p1", "p2"], frames, labels)
    return frames, truth


def hallway_bathroom_door() -> tuple[list[list[dict]], dict]:
    """One senior at a bathroom door seen by hallway + doorway camera."""
    emb_hall = [0.85, 0.15, 0.00]
    emb_door = [0.83, 0.17, 0.00]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(10):
        fx = 3000 + i * 150
        det_id = f"det-hbd-hall-{i}"
        frames.append([_obs("cam-hall", i, i, fx, 8000, emb_hall, detection_id=det_id)])
        labels.append(["p1"])

    for i in range(10):
        step = 10 + i
        fx = 4500 + i * 20
        det_h = f"det-hbd-hall-{step}"
        det_d = f"det-hbd-door-{step}"
        frames.append(
            [
                _obs("cam-hall", step, step, fx, 8000, emb_hall, detection_id=det_h),
                _obs("cam-door", step, step, fx + 30, 8050, emb_door, detection_id=det_d),
            ]
        )
        labels.append(["p1", "p1"])

    for i in range(8):
        step = 20 + i
        frames.append([])
        labels.append([])

    for i in range(10):
        step = 28 + i
        fx = 4700 - i * 150
        det_id = f"det-hbd-hall-ret-{i}"
        frames.append([_obs("cam-hall", step, step, fx, 8000, emb_hall, detection_id=det_id)])
        labels.append(["p1"])

    truth = _make_truth_with_events(
        ["p1"],
        frames,
        labels,
        events=[{"type": "exit", "person": "p1", "at_step": 20, "duration_frames": 8}],
    )
    return frames, truth


# ── Identity continuity fixtures ────────────────────────────────────────


def _mk_face(
    person_id: str,
    confidence: float,
    *,
    camera_id: str,
    detection_id: str,
    step: int,
    quality: float = 0.9,
) -> dict:
    return {
        "person_id": person_id,
        "confidence": confidence,
        "quality": quality,
        "detection_id": detection_id,
        "camera_id": camera_id,
        "captured_at_iso": (BASE_TIME + timedelta(seconds=step * FRAME_INTERVAL_S)).isoformat(),
    }


def _lerp(a: list[float], b: list[float], t: float) -> list[float]:
    return [round(a[i] + (b[i] - a[i]) * t, 6) for i in range(len(a))]


def single_camera_turn() -> tuple[list[list[dict]], dict]:
    """One person, one uncalibrated camera, turning from front to back.

    Frames 0-9:   front-facing, face anchor present (high confidence).
    Frames 10-17: turning -- face anchor weakens (confidence 0.75->0.55),
                   body embedding drifts from front to side.
    Frames 18-29: EMPTY -- person occluded / detector dropped.
                   Gap length = 12 frames x 0.5 s = 6 s > ph_close_grace_s (5 s),
                   so the PH closes and a respawn is forced.  This demonstrates
                   PH churn (close + new spawn) in addition to resolver demotion.
    Frames 30-35: side-on -- no face anchor, embedding drifts side->back.
                   Spawns a new PH (UNKNOWN, no face anchor at spawn).
    Frames 36-49: back-on -- no face anchor, embedding = back.

    Total: 50 frames.  calibrated=False throughout (home camera scenario).

    Without revival/sticky maintenance this produces 2 distinct PHs and UNKNOWN-after-known.
    With revival + sticky maintenance this should produce 1 PH with
    identity alice held throughout.
    """
    emb_front = [0.95, 0.05, 0.00]
    emb_side = [0.70, 0.28, 0.02]
    emb_back = [0.50, 0.45, 0.05]
    camera_id = "cam-turn"
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    # Frames 0-9: front-facing with face anchor
    for i in range(10):
        fx = 3000 + i * 100
        det_id = f"det-turn-front-{i}"
        frames.append(
            [
                _obs(
                    camera_id,
                    i,
                    i,
                    fx,
                    5000,
                    emb_front,
                    calibrated=False,
                    face_anchor=_mk_face(
                        "alice",
                        0.85,
                        camera_id=camera_id,
                        detection_id=det_id,
                        step=i,
                    ),
                    detection_id=det_id,
                    orientation=ORI_FRONT,
                    orientation_confidence=0.85,
                )
            ]
        )
        labels.append(["alice"])

    # Frames 10-17: turning -- face weakens, embedding drifts front->side
    for i in range(8):
        step = 10 + i
        fx = 4000 + i * 100
        t = (i + 1) / 8.0
        emb = _lerp(emb_front, emb_side, t)
        conf = round(0.75 - i * 0.03125, 2)
        det_id = f"det-turn-mid-{i}"
        frames.append(
            [
                _obs(
                    camera_id,
                    step,
                    step,
                    fx,
                    5000,
                    emb,
                    calibrated=False,
                    face_anchor=_mk_face(
                        "alice",
                        conf,
                        camera_id=camera_id,
                        detection_id=det_id,
                        step=step,
                    ),
                    detection_id=det_id,
                    orientation=ORI_LEFT,
                    orientation_confidence=round(0.5 + i * 0.05, 2),
                )
            ]
        )
        labels.append(["alice"])

    # Frames 18-29: EMPTY -- occlusion gap
    for _ in range(12):
        frames.append([])
        labels.append([])

    # Frames 30-35: side-on -- no face anchor
    for i in range(6):
        step = 30 + i
        fx = 4800 + i * 100
        t = i / 5.0 if i > 0 else 0.0
        emb = _lerp(emb_side, emb_back, t)
        det_id = f"det-turn-side-{i}"
        frames.append(
            [
                _obs(
                    camera_id,
                    step,
                    step,
                    fx,
                    5000,
                    emb,
                    calibrated=False,
                    detection_id=det_id,
                    orientation=ORI_LEFT,
                    orientation_confidence=0.4,
                )
            ]
        )
        labels.append(["alice"])

    # Frames 36-49: back-on -- no face anchor
    for i in range(14):
        step = 36 + i
        fx = 5400 + i * 100
        det_id = f"det-turn-back-{i}"
        frames.append(
            [
                _obs(
                    camera_id,
                    step,
                    step,
                    fx,
                    5000,
                    emb_back,
                    calibrated=False,
                    detection_id=det_id,
                    orientation=ORI_BACK,
                    orientation_confidence=0.8,
                )
            ]
        )
        labels.append(["alice"])

    truth = _make_truth_with_events(
        ["alice"],
        frames,
        labels,
        events=[
            {"type": "exit", "person": "alice", "at_step": 18, "duration_frames": 12},
        ],
    )
    return frames, truth


def cross_camera_handoff() -> tuple[list[list[dict]], dict]:
    """One person, two uncalibrated cameras, non-overlapping in time.

    Camera A frames 0-15, camera B frames 16-30.  Face anchor on camera A
    only at frames 0-4.  Same identity embedding (with slight view change).
    """
    emb_a = [0.85, 0.15, 0.00]
    emb_b = [0.80, 0.20, 0.00]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    # Camera A: frames 0-15
    for i in range(16):
        fx = 3000 + i * 200
        fa = (
            _mk_face(
                "alice",
                0.82,
                camera_id="cam-handoff-a",
                detection_id=f"det-handoff-a-{i}",
                step=i,
            )
            if i <= 4
            else None
        )
        det_id = f"det-handoff-a-{i}"
        frames.append(
            [
                _obs(
                    "cam-handoff-a",
                    i,
                    i,
                    fx,
                    5000,
                    emb_a,
                    calibrated=False,
                    face_anchor=fa,
                    detection_id=det_id,
                    orientation=ORI_FRONT if i <= 4 else ORI_UNKNOWN,
                    orientation_confidence=0.8 if i <= 4 else 0.0,
                )
            ]
        )
        labels.append(["alice"])

    # Gap: 32 empty frames (16 s > ph_close_grace_s = 15 s)
    gap_start = 16
    gap_frames = 32
    for _ in range(gap_frames):
        frames.append([])
        labels.append([])

    # Camera B: frames after gap
    cam_b_start = gap_start + gap_frames
    for i in range(15):
        step = cam_b_start + i
        fx = 8000 + i * 200
        det_id = f"det-handoff-b-{i}"
        frames.append(
            [
                _obs(
                    "cam-handoff-b",
                    i,
                    step,
                    fx,
                    5000,
                    emb_b,
                    calibrated=False,
                    detection_id=det_id,
                    orientation=ORI_UNKNOWN,
                )
            ]
        )
        labels.append(["alice"])

    truth = _make_truth_with_events(
        ["alice"],
        frames,
        labels,
        events=[
            {"type": "exit", "person": "alice", "at_step": 16, "duration_frames": 32},
            {
                "type": "handoff",
                "person": "alice",
                "from_camera": "cam-handoff-a",
                "to_camera": "cam-handoff-b",
                "at_step": cam_b_start,
            },
        ],
    )
    return frames, truth


def two_people_one_room() -> tuple[list[list[dict]], dict]:
    """Two enrolled people, one uncalibrated camera, crossing paths.

    Person A (alice): embedding close to [1.0, 0.0, 0.0], face anchor at start.
    Person B (bob):   embedding close to [0.0, 0.95, 0.05], face anchor at start.

    Frames 0-15:  both visible, approaching each other.
    Frames 16-25: crossing paths (positions converge then diverge).
    """
    emb_a = [0.98, 0.01, 0.01]
    emb_b = [0.01, 0.96, 0.03]
    camera_id = "cam-cross"
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(26):
        fxa = 3000 + i * 200
        fxb = 8000 - i * 200

        det_a = f"det-cross-a-{i}"
        det_b = f"det-cross-b-{i}"

        fa_a = (
            _mk_face("alice", 0.84, camera_id=camera_id, detection_id=det_a, step=i)
            if i <= 4
            else None
        )
        fa_b = (
            _mk_face("bob", 0.83, camera_id=camera_id, detection_id=det_b, step=i)
            if i <= 4
            else None
        )

        frames.append(
            [
                _obs(
                    camera_id,
                    i,
                    i,
                    fxa,
                    5000,
                    emb_a,
                    calibrated=False,
                    face_anchor=fa_a,
                    detection_id=det_a,
                    orientation=ORI_FRONT if i <= 4 else ORI_UNKNOWN,
                    orientation_confidence=0.8 if i <= 4 else 0.0,
                ),
                _obs(
                    camera_id,
                    i,
                    i,
                    fxb,
                    5000,
                    emb_b,
                    calibrated=False,
                    face_anchor=fa_b,
                    detection_id=det_b,
                    orientation=ORI_FRONT if i <= 4 else ORI_UNKNOWN,
                    orientation_confidence=0.8 if i <= 4 else 0.0,
                ),
            ]
        )
        labels.append(["alice", "bob"])

    truth = _make_truth(["alice", "bob"], frames, labels)
    return frames, truth


def resident_plus_stranger() -> tuple[list[list[dict]], dict]:
    """One enrolled resident + one unenrolled stranger, one camera, crossing paths.

    Resident (alice): has face anchor at frames 0-4.
    Stranger:          never produces a face anchor (only no-face frames).

    This is the clinical guardrail for the favor-continuity bias:
    later identity changes must never transfer the resident's identity onto
    the stranger's track.
    """
    emb_resident = [0.90, 0.10, 0.00]
    emb_stranger = [0.30, 0.50, 0.20]
    camera_id = "cam-guard"
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(30):
        fxa = 3000 + i * 150
        fxb = 7500 - i * 150

        det_res = f"det-guard-res-{i}"
        det_str = f"det-guard-str-{i}"

        fa_resident = (
            _mk_face("alice", 0.86, camera_id=camera_id, detection_id=det_res, step=i)
            if i <= 4
            else None
        )

        frames.append(
            [
                _obs(
                    camera_id,
                    i,
                    i,
                    fxa,
                    5000,
                    emb_resident,
                    calibrated=False,
                    face_anchor=fa_resident,
                    detection_id=det_res,
                    orientation=ORI_FRONT if i <= 4 else ORI_UNKNOWN,
                    orientation_confidence=0.8 if i <= 4 else 0.0,
                ),
                _obs(
                    camera_id,
                    i,
                    i,
                    fxb,
                    5000,
                    emb_stranger,
                    calibrated=False,
                    detection_id=det_str,
                    orientation=ORI_UNKNOWN,
                ),
            ]
        )
        labels.append(["alice", "stranger"])

    truth = _make_truth(["alice", "stranger"], frames, labels)
    return frames, truth


# ── Sweep fixtures ──────────────────────────────────────────────────────


def uncalibrated_pacing() -> tuple[list[list[dict]], dict]:
    """One person, one uncalibrated home camera, pacing back and forth.

    Designed to stress the revival path: person disappears for 3 frames
    (< ph_close_grace_s=5s at 0.5s interval → 1.5s gap) and reappears,
    then disappears for 12 frames (6s > grace → PH must close+respawn).

    Frames  0-14:  walking forward (x 3000→5800)
    Frames 15-17:  EMPTY (3 frames = 1.5s < grace; PH coasts)
    Frames 18-29:  walking back (x 5600→3400)
    Frames 30-41:  EMPTY (12 frames = 6s > grace; PH closes)
    Frames 42-55:  walking forward again (x 3000→5600)

    calibrated=False throughout.
    """
    emb = [0.80, 0.18, 0.02]
    camera_id = "cam-pace"
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    # Phase 1: walk forward
    for i in range(15):
        step = i
        fx = 3000 + i * 190
        det_id = f"det-pace-fwd-{i}"
        frames.append(
            [_obs(camera_id, i, step, fx, 5000, emb, calibrated=False, detection_id=det_id)]
        )
        labels.append(["p1"])

    # Short gap: 3 empty frames
    for _ in range(3):
        frames.append([])
        labels.append([])

    # Phase 2: walk back
    for i in range(12):
        step = 18 + i
        fx = 5600 - i * 183
        det_id = f"det-pace-bk-{i}"
        frames.append(
            [_obs(camera_id, step, step, fx, 5000, emb, calibrated=False, detection_id=det_id)]
        )
        labels.append(["p1"])

    # Long gap: 12 empty frames (forces PH close)
    for _ in range(12):
        frames.append([])
        labels.append([])

    # Phase 3: walk forward again (new PH or revival)
    for i in range(14):
        step = 42 + i
        fx = 3000 + i * 190
        det_id = f"det-pace-fwd2-{i}"
        frames.append(
            [_obs(camera_id, step, step, fx, 5000, emb, calibrated=False, detection_id=det_id)]
        )
        labels.append(["p1"])

    truth = _make_truth_with_events(
        ["p1"],
        frames,
        labels,
        events=[
            {"type": "gap", "person": "p1", "at_step": 15, "duration_frames": 3},
            {"type": "exit", "person": "p1", "at_step": 30, "duration_frames": 12},
        ],
    )
    return frames, truth


def uncalibrated_two_people_home() -> tuple[list[list[dict]], dict]:
    """Two people on a single uncalibrated home camera.

    Alice: enrolled, has face anchor at start. Bob: unenrolled, no face anchor.
    They stay on distinct sides of the room (no crossing), testing that
    two PHs remain distinct throughout on an uncalibrated camera.

    Frames 0-24: alice left side, bob right side. Both always visible.
    """
    emb_alice = [0.92, 0.07, 0.01]
    emb_bob = [0.05, 0.88, 0.07]
    camera_id = "cam-home-2p"
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(25):
        fxa = 2000 + i * 100  # alice: 2m → 4.4m
        fxb = 8000 + i * 80  # bob: 8m → 9.92m (well separated)

        det_a = f"det-home2p-a-{i}"
        det_b = f"det-home2p-b-{i}"

        fa_a = (
            _mk_face("alice", 0.87, camera_id=camera_id, detection_id=det_a, step=i)
            if i <= 3
            else None
        )

        frames.append(
            [
                _obs(
                    camera_id,
                    i,
                    i,
                    fxa,
                    5000,
                    emb_alice,
                    calibrated=False,
                    face_anchor=fa_a,
                    detection_id=det_a,
                    orientation=ORI_FRONT if i <= 3 else ORI_UNKNOWN,
                    orientation_confidence=0.85 if i <= 3 else 0.0,
                ),
                _obs(
                    camera_id,
                    i,
                    i,
                    fxb,
                    5000,
                    emb_bob,
                    calibrated=False,
                    detection_id=det_b,
                    orientation=ORI_UNKNOWN,
                ),
            ]
        )
        labels.append(["alice", "bob"])

    truth = _make_truth(["alice", "bob"], frames, labels)
    return frames, truth


def mixed_calibration_entry() -> tuple[list[list[dict]], dict]:
    """One person entering a home via a calibrated entry camera, then moving through
    an uncalibrated room camera.

    cam-entry (calibrated): frames 0-9. Person enters, floor point valid.
    cam-room  (uncalibrated): frames 10-24. Person moves through room.

    Tests that the tracker can associate across calibration boundaries.
    """
    emb = [0.88, 0.11, 0.01]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    # Entry camera (calibrated)
    for i in range(10):
        step = i
        fx = 1000 + i * 300  # 1m → 3.7m in entry zone
        det_id = f"det-mixed-entry-{i}"
        frames.append(
            [
                _obs(
                    "cam-entry",
                    i,
                    step,
                    fx,
                    2000,
                    emb,
                    calibrated=True,
                    detection_id=det_id,
                    face_anchor=(
                        _mk_face(
                            "alice",
                            0.82,
                            camera_id="cam-entry",
                            detection_id=det_id,
                            step=step,
                        )
                        if i <= 2
                        else None
                    ),
                    orientation=ORI_FRONT if i <= 2 else ORI_UNKNOWN,
                    orientation_confidence=0.8 if i <= 2 else 0.0,
                )
            ]
        )
        labels.append(["alice"])

    # Room camera (uncalibrated)
    for i in range(15):
        step = 10 + i
        det_id = f"det-mixed-room-{i}"
        frames.append(
            [
                _obs(
                    "cam-room",
                    i,
                    step,
                    5000 + i * 200,
                    5000,
                    emb,
                    calibrated=False,
                    detection_id=det_id,
                    orientation=ORI_UNKNOWN,
                )
            ]
        )
        labels.append(["alice"])

    truth = _make_truth_with_events(
        ["alice"],
        frames,
        labels,
        events=[
            {
                "type": "handoff",
                "person": "alice",
                "from_camera": "cam-entry",
                "to_camera": "cam-room",
                "at_step": 10,
            },
        ],
    )
    return frames, truth


def stationary_two_camera() -> tuple[list[list[dict]], dict]:
    """Person standing still seen by two cameras with different calibration offsets.

    cam-A: systematic +0.3 m offset in x  →  observations at (8300, 8000) mm
    cam-B: systematic -0.2 m offset in x  →  observations at (7800, 8000) mm

    Both calibrated with floor_residual_m matching their offset magnitude so the
    bias floor in dedup._build_representative carries the right uncertainty.
    Inter-camera distance = 500 mm = 0.5 m < dedup_max_distance_m (0.6 m), so
    every frame the two observations collapse into one fused representative.

    Phase 1 (steps 0-29): both cameras present.
    Phase 2 (steps 30-39): cam-A only (cam-B dropout).

    Truth sidecar extras:
        truth_position_mm       [8000, 8000] — true floor position
        cam_b_dropout_at_step   30
    """
    emb = [0.90, 0.08, 0.02]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(30):
        det_a = f"det-stat-a-{i}"
        det_b = f"det-stat-b-{i}"
        frames.append(
            [
                _obs(
                    "cam-A",
                    i,
                    i,
                    8300,
                    8000,
                    emb,
                    detection_id=det_a,
                    quality=0.85,
                    floor_cov_random=[0.0025, 0.0, 0.0, 0.0025],
                    floor_residual_m=0.30,
                ),
                _obs(
                    "cam-B",
                    i,
                    i,
                    7800,
                    8000,
                    emb,
                    detection_id=det_b,
                    quality=0.80,
                    floor_cov_random=[0.0025, 0.0, 0.0, 0.0025],
                    floor_residual_m=0.20,
                ),
            ]
        )
        labels.append(["p1", "p1"])

    for i in range(10):
        step = 30 + i
        det_a = f"det-stat-a-{step}"
        frames.append(
            [
                _obs(
                    "cam-A",
                    step,
                    step,
                    8300,
                    8000,
                    emb,
                    detection_id=det_a,
                    quality=0.85,
                    floor_cov_random=[0.0025, 0.0, 0.0, 0.0025],
                    floor_residual_m=0.30,
                ),
            ]
        )
        labels.append(["p1"])

    truth = _make_truth(["p1"], frames, labels)
    truth["truth_position_mm"] = [8000, 8000]
    truth["cam_b_dropout_at_step"] = 30
    return frames, truth


def slow_shuffle() -> tuple[list[list[dict]], dict]:
    """Person walking at 0.3 m/s along the x-axis; single calibrated camera.

    Frame interval = 0.5 s, so each step advances 150 mm in x.
    Walk from (3000, 5000) mm to (3000 + 39*150, 5000) = (8850, 5000) mm.

    Truth sidecar extras:
        truth_trajectory_mm   [[x_mm, y_mm], ...]  per frame
        truth_speed_m_s       0.30
    """
    emb = [0.85, 0.10, 0.05]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []
    trajectory_mm: list[list[int]] = []

    step_mm = 150  # 0.3 m/s * 0.5 s
    start_x_mm = 3000
    y_mm = 5000

    for i in range(40):
        fx = start_x_mm + i * step_mm
        det_id = f"det-shuffle-{i}"
        frames.append(
            [
                _obs(
                    "cam-walk",
                    i,
                    i,
                    fx,
                    y_mm,
                    emb,
                    detection_id=det_id,
                    quality=0.80,
                    floor_cov_random=[0.0064, 0.0, 0.0, 0.0064],
                    floor_residual_m=0.05,
                ),
            ]
        )
        labels.append(["p1"])
        trajectory_mm.append([fx, y_mm])

    truth = _make_truth(["p1"], frames, labels)
    truth["truth_trajectory_mm"] = trajectory_mm
    truth["truth_speed_m_s"] = 0.30
    return frames, truth


def oblique_single_camera() -> tuple[list[list[dict]], dict]:
    """Person at a fixed position viewed by a single oblique camera.

    The camera views mostly along the x-axis (large Jacobian in x), producing an
    elongated error ellipse: sigma_x = 0.4 m, sigma_y = 0.1 m.

    floor_cov_random = [0.16, 0.0, 0.0, 0.01]  (row-major 2x2, m²)

    After _finalize_singleton adds the bias floor and the Kalman applies the
    anisotropic R, the PH posterior covariance eigen-ratio should be ≫ 1.

    Truth sidecar extras:
        truth_position_mm   [8000, 8000]
    """
    emb = [0.80, 0.15, 0.05]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    for i in range(40):
        det_id = f"det-oblique-{i}"
        frames.append(
            [
                _obs(
                    "cam-oblique",
                    i,
                    i,
                    8000,
                    8000,
                    emb,
                    detection_id=det_id,
                    quality=0.75,
                    floor_cov_random=[0.16, 0.0, 0.0, 0.01],
                    floor_residual_m=0.05,
                ),
            ]
        )
        labels.append(["p1"])

    truth = _make_truth(["p1"], frames, labels)
    truth["truth_position_mm"] = [8000, 8000]
    return frames, truth


def posture_disagreement_four_camera() -> tuple[list[list[dict]], dict]:
    """Four cameras with conflicting posture evidence.

    Three side cameras (LEFT, RIGHT, BACK) see a sitting person.
    One frontal camera (FRONT) sees a standing person with high keypoint confidence.
    Geometry-aware posture fusion should output 'sitting' because side views carry
    higher view_weight (they better reveal height-relative posture cues).

    Truth sidecar extras:
        posture_truth               "sitting"
        per_camera_posture_scores   dict[camera_id, {lying, sitting, sw, kp_conf}]
    """
    emb = [0.85, 0.10, 0.05]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []

    # Side cameras have LOW keypoint confidence (kp=0.30) so that, under equal
    # view weights, the high-kp front camera's strong standing signal dominates
    # and sw wins.  Only when the frontal view-weight penalty (0.64) is applied
    # does sitting's three-camera sum re-take the lead.  This makes the gate
    # falsifiable: a constant view_weight=1.0 would flip the result to "standing".
    cameras: list[tuple[str, int, float]] = [
        ("cam-left", ORI_LEFT, 0.30),
        ("cam-right", ORI_RIGHT, 0.30),
        ("cam-back", ORI_BACK, 0.30),
        ("cam-front", ORI_FRONT, 0.95),
    ]

    for i in range(30):
        step_obs: list[dict] = []
        step_labels: list[str] = []
        for cam_id, ori, quality in cameras:
            det_id = f"det-posture-{cam_id}-{i}"
            step_obs.append(
                _obs(
                    cam_id,
                    i,
                    i,
                    8000,
                    8000,
                    emb,
                    detection_id=det_id,
                    quality=quality,
                    orientation=ori,
                    orientation_confidence=0.9,
                    floor_cov_random=[0.04, 0.0, 0.0, 0.04],
                )
            )
            step_labels.append("p1")
        frames.append(step_obs)
        labels.append(step_labels)

    truth = _make_truth(["p1"], frames, labels)
    truth["posture_truth"] = "sitting"
    # Scores designed so equal-weight fusion picks "standing_walking" (front camera
    # dominates via kp=0.95), but geometry-aware weighting (front view_weight=0.64,
    # side view_weight=1.0) tips the result back to "sitting".
    # Equal-weight check: sw=0.522 > sitting=0.452  (test would fail without weighting)
    # Geometry-weight check: sitting=0.511 > sw=0.459  (test passes with weighting)
    truth["per_camera_posture_scores"] = {
        "cam-left": {
            "lying": 0.05,
            "sitting": 0.85,
            "standing_walking": 0.10,
            "keypoint_confidence": 0.30,
        },
        "cam-right": {
            "lying": 0.06,
            "sitting": 0.82,
            "standing_walking": 0.12,
            "keypoint_confidence": 0.30,
        },
        "cam-back": {
            "lying": 0.05,
            "sitting": 0.80,
            "standing_walking": 0.15,
            "keypoint_confidence": 0.30,
        },
        "cam-front": {
            "lying": 0.00,
            "sitting": 0.10,
            "standing_walking": 0.90,
            "keypoint_confidence": 0.95,
        },
    }
    return frames, truth


def moving_then_stop() -> tuple[list[list[dict]], dict]:
    """Walk then stop: tests ZUPT engages only after the stop, lag bounded during walk.

    Steps 0-19:  walking at 0.3 m/s (150 mm/step in x), single camera.
    Steps 20-59: stationary at (6000, 5000) mm.
    ZUPT requires zupt_consecutive_frames=5 stationary frames; by step 25 it fires.

    Truth sidecar extras:
        truth_trajectory_mm   [[x_mm, y_mm], ...]  per frame (60 entries)
        walk_end_step         20
        truth_stop_pos_mm     [6000, 5000]
    """
    emb = [0.87, 0.10, 0.03]
    frames: list[list[dict]] = []
    labels: list[list[str]] = []
    trajectory_mm: list[list[int]] = []

    step_mm = 150
    start_x_mm = 3000
    stop_x_mm = 6000
    y_mm = 5000
    walk_frames = 20
    stop_frames = 40

    for i in range(walk_frames):
        fx = start_x_mm + i * step_mm
        det_id = f"det-move-walk-{i}"
        frames.append(
            [
                _obs(
                    "cam-move",
                    i,
                    i,
                    fx,
                    y_mm,
                    emb,
                    detection_id=det_id,
                    quality=0.80,
                    floor_cov_random=[0.0064, 0.0, 0.0, 0.0064],
                    floor_residual_m=0.05,
                ),
            ]
        )
        labels.append(["p1"])
        trajectory_mm.append([fx, y_mm])

    for i in range(stop_frames):
        step = walk_frames + i
        det_id = f"det-move-stop-{i}"
        frames.append(
            [
                _obs(
                    "cam-move",
                    step,
                    step,
                    stop_x_mm,
                    y_mm,
                    emb,
                    detection_id=det_id,
                    quality=0.80,
                    floor_cov_random=[0.0064, 0.0, 0.0, 0.0064],
                    floor_residual_m=0.05,
                ),
            ]
        )
        labels.append(["p1"])
        trajectory_mm.append([stop_x_mm, y_mm])

    truth = _make_truth(["p1"], frames, labels)
    truth["truth_trajectory_mm"] = trajectory_mm
    truth["walk_end_step"] = walk_frames
    truth["truth_stop_pos_mm"] = [stop_x_mm, y_mm]
    return frames, truth


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    fixtures: list[tuple[str, tuple[list[list[dict]], dict]]] = [
        ("two_cameras_one_room", two_cameras_one_room()),
        ("two_rooms_two_people", two_rooms_two_people()),
        ("hallway_bathroom_door", hallway_bathroom_door()),
        ("single_camera_turn", single_camera_turn()),
        ("cross_camera_handoff", cross_camera_handoff()),
        ("two_people_one_room", two_people_one_room()),
        ("resident_plus_stranger", resident_plus_stranger()),
        ("uncalibrated_pacing", uncalibrated_pacing()),
        ("uncalibrated_two_people_home", uncalibrated_two_people_home()),
        ("mixed_calibration_entry", mixed_calibration_entry()),
        # M09 acceptance fixtures
        ("stationary_two_camera", stationary_two_camera()),
        ("slow_shuffle", slow_shuffle()),
        ("oblique_single_camera", oblique_single_camera()),
        ("posture_disagreement_four_camera", posture_disagreement_four_camera()),
        ("moving_then_stop", moving_then_stop()),
    ]

    for name, (frames, truth) in fixtures:
        _write(FIXTURES_DIR / f"{name}.bin", frames, truth)

    print("done")


if __name__ == "__main__":
    sys.exit(main())
