"""Synthesize deterministic WorldObservation replay fixtures for integration tests.

Format: length-prefixed JSON binary. Each chunk is one "frame step" (one call
to WorldTracker.step()), encoded as a JSON array of observation dicts:

    [u32 BE length][JSON bytes] [u32 BE length][JSON bytes] ...

Each observation dict contains:
    camera_id           str
    frame_index         int
    captured_at_iso     str   (ISO-8601 with timezone)
    floor_x_mm          int
    floor_y_mm          int
    embedding           list[float]
    detection_confidence float
    bbox                dict  x_min/y_min/y_max/y_max
    calibrated          bool  (default True; M1 added)
    face_anchor         dict | None  (optional; M1 added)

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
    detection_id: str | None = None,
    detection_confidence: float = 0.92,
    quality: float = 0.5,
) -> dict:
    import uuid as _uuid

    obs_time = BASE_TIME + timedelta(seconds=step * FRAME_INTERVAL_S)
    result: dict = {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "captured_at_iso": obs_time.isoformat(),
        "floor_x_mm": floor_x_mm,
        "floor_y_mm": floor_y_mm,
        "embedding": embedding,
        "detection_confidence": detection_confidence,
        "bbox": {"x_min": 100, "y_min": 100, "x_max": 300, "y_max": 400},
        "detection_id": detection_id if detection_id else str(_uuid.uuid4()),
        "quality": quality,
        "calibrated": calibrated,
    }
    if face_anchor is not None:
        fa = dict(face_anchor)
        if "captured_at_iso" not in fa:
            fa["captured_at_iso"] = obs_time.isoformat()
        result["face_anchor"] = fa
    return result


def _write(path: Path, frames: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for observations in frames:
            data = json.dumps(observations).encode()
            f.write(struct.pack(">I", len(data)))
            f.write(data)
    size_kb = path.stat().st_size // 1024
    print(f"wrote {path.name}: {len(frames)} steps, {size_kb} KB")


# ── Existing fixtures (unchanged behavior) ──────────────────────────────


def two_cameras_one_room() -> list[list[dict]]:
    """One person walking under two overlapping cameras in one room."""
    emb_a = [0.90, 0.10, 0.00]
    emb_a2 = [0.88, 0.12, 0.00]
    frames: list[list[dict]] = []

    for i in range(10):
        fx = 3000 + i * 200
        frames.append([_obs("cam-1", i, i, fx, 5000, emb_a)])

    for i in range(10):
        step = 10 + i
        fx = 5000 + i * 200
        frames.append(
            [
                _obs("cam-1", step, step, fx, 5000, emb_a),
                _obs("cam-2", step, step, fx, 5000, emb_a2),
            ]
        )

    for i in range(10):
        step = 20 + i
        fx = 7000 + i * 200
        frames.append([_obs("cam-2", step, step, fx, 5000, emb_a2)])

    return frames


def two_rooms_two_people() -> list[list[dict]]:
    """Two people in two non-overlapping rooms that never merge."""
    emb_a = [1.00, 0.00, 0.00]
    emb_b = [0.00, 1.00, 0.00]
    frames: list[list[dict]] = []

    for i in range(20):
        step = i
        fxa = 2000 + i * 50
        fxb = 15000 + i * 50
        frames.append(
            [
                _obs("cam-1", step, step, fxa, 2000, emb_a),
                _obs("cam-2", step, step, fxb, 15000, emb_b),
            ]
        )

    return frames


def hallway_bathroom_door() -> list[list[dict]]:
    """One senior at a bathroom door seen by hallway + doorway camera."""
    emb_hall = [0.85, 0.15, 0.00]
    emb_door = [0.83, 0.17, 0.00]
    frames: list[list[dict]] = []

    for i in range(10):
        fx = 3000 + i * 150
        frames.append([_obs("cam-hall", i, i, fx, 8000, emb_hall)])

    for i in range(10):
        step = 10 + i
        fx = 4500 + i * 20
        frames.append(
            [
                _obs("cam-hall", step, step, fx, 8000, emb_hall),
                _obs("cam-door", step, step, fx + 30, 8050, emb_door),
            ]
        )

    for _ in range(8):
        frames.append([])

    for i in range(10):
        step = 28 + i
        fx = 4700 - i * 150
        frames.append([_obs("cam-hall", step, step, fx, 8000, emb_hall)])

    return frames


# ── M1 new fixtures ─────────────────────────────────────────────────────


def _mk_face(
    person_id: str,
    confidence: float,
    *,
    camera_id: str,
    detection_id: str,
    step: int,
    quality: float = 0.9,
) -> dict:
    """Build a face_anchor dict for the fixture."""
    return {
        "person_id": person_id,
        "confidence": confidence,
        "quality": quality,
        "detection_id": detection_id,
        "camera_id": camera_id,
        "captured_at_iso": (BASE_TIME + timedelta(seconds=step * FRAME_INTERVAL_S)).isoformat(),
    }


def _lerp(a: list[float], b: list[float], t: float) -> list[float]:
    """Linear interpolation between two embedding vectors."""
    return [round(a[i] + (b[i] - a[i]) * t, 6) for i in range(len(a))]


def single_camera_turn() -> list[list[dict]]:
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

    Without M2 flags this produces 2 distinct PHs and UNKNOWN-after-known.
    With M2 revival + sticky maintenance this should produce 1 PH with
    identity alice held throughout.
    """
    emb_front = [0.95, 0.05, 0.00]
    emb_side = [0.70, 0.28, 0.02]
    emb_back = [0.50, 0.45, 0.05]
    camera_id = "cam-turn"
    frames: list[list[dict]] = []

    # Frames 0-9: front-facing with face anchor
    for i in range(10):
        fx = 3000 + i * 100
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
                        detection_id=f"det-turn-front-{i}",
                        step=i,
                    ),
                    detection_id=f"det-turn-front-{i}",
                )
            ]
        )

    # Frames 10-17: turning -- face weakens, embedding drifts front->side
    for i in range(8):
        step = 10 + i
        fx = 4000 + i * 100
        t = (i + 1) / 8.0  # 0.125 -> 1.0
        emb = _lerp(emb_front, emb_side, t)
        conf = round(0.75 - i * 0.03125, 2)  # 0.75 -> 0.53
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
                        detection_id=f"det-turn-mid-{i}",
                        step=step,
                    ),
                    detection_id=f"det-turn-mid-{i}",
                )
            ]
        )

    # Frames 18-29: EMPTY -- occlusion gap (12 frames x 0.5 s = 6 s).
    # This exceeds ph_close_grace_s (5 s), forcing PH close + respawn.
    for i in range(12):
        step = 18 + i
        frames.append([])

    # Frames 30-35: side-on -- no face anchor, embedding drifts side->back
    for i in range(6):
        step = 30 + i
        fx = 4800 + i * 100
        t = i / 5.0 if i > 0 else 0.0
        emb = _lerp(emb_side, emb_back, t)
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
                    detection_id=f"det-turn-side-{i}",
                )
            ]
        )

    # Frames 36-49: back-on -- no face anchor, embedding = back
    for i in range(14):
        step = 36 + i
        fx = 5400 + i * 100
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
                    detection_id=f"det-turn-back-{i}",
                )
            ]
        )

    return frames


def cross_camera_handoff() -> list[list[dict]]:
    """One person, two uncalibrated cameras, non-overlapping in time.

    Camera A frames 0-15, camera B frames 16-30.  Face anchor on camera A
    only at frames 0-4.  Same identity embedding (with slight view change).
    """
    emb_a = [0.85, 0.15, 0.00]
    emb_b = [0.80, 0.20, 0.00]  # slightly different (view change between cams)
    frames: list[list[dict]] = []

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
                    detection_id=f"det-handoff-a-{i}",
                )
            ]
        )

    # Gap: 32 empty frames (16 s > ph_close_grace_s = 15 s).
    # This ensures the PH closes before camera B starts, demonstrating
    # the cross-camera disconnect that M5 must fix.
    gap_start = 16
    gap_frames = 32
    for _ in range(gap_frames):
        frames.append([])

    # Camera B: frames after gap
    cam_b_start = gap_start + gap_frames
    for i in range(15):
        step = cam_b_start + i
        fx = 8000 + i * 200
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
                    detection_id=f"det-handoff-b-{i}",
                )
            ]
        )

    return frames


def two_people_one_room() -> list[list[dict]]:
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

    for i in range(26):
        # Person A: starts at x=3m, moves to x=8m
        fxa = 3000 + i * 200
        # Person B: starts at x=8m, moves to x=3m
        fxb = 8000 - i * 200

        fa_a = (
            _mk_face(
                "alice",
                0.84,
                camera_id=camera_id,
                detection_id=f"det-cross-a-{i}",
                step=i,
            )
            if i <= 4
            else None
        )
        fa_b = (
            _mk_face(
                "bob",
                0.83,
                camera_id=camera_id,
                detection_id=f"det-cross-b-{i}",
                step=i,
            )
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
                    detection_id=f"det-cross-a-{i}",
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
                    detection_id=f"det-cross-b-{i}",
                ),
            ]
        )

    return frames


def resident_plus_stranger() -> list[list[dict]]:
    """One enrolled resident + one unenrolled stranger, one camera, crossing paths.

    Resident (alice): has face anchor at frames 0-4.
    Stranger:          never produces a face anchor (only no-face frames).

    This is the clinical guardrail for the favor-continuity bias:
    later milestones must never transfer the resident's identity onto
    the stranger's track.
    """
    emb_resident = [0.90, 0.10, 0.00]
    emb_stranger = [0.30, 0.50, 0.20]  # far from resident in embedding space
    camera_id = "cam-guard"
    frames: list[list[dict]] = []

    for i in range(30):
        fxa = 3000 + i * 150
        fxb = 7500 - i * 150

        fa_resident = (
            _mk_face(
                "alice",
                0.86,
                camera_id=camera_id,
                detection_id=f"det-guard-res-{i}",
                step=i,
            )
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
                    detection_id=f"det-guard-res-{i}",
                ),
                _obs(
                    camera_id,
                    i,
                    i,
                    fxb,
                    5000,
                    emb_stranger,
                    calibrated=False,
                    detection_id=f"det-guard-str-{i}",
                ),
            ]
        )

    return frames


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _write(FIXTURES_DIR / "two_cameras_one_room.bin", two_cameras_one_room())
    _write(FIXTURES_DIR / "two_rooms_two_people.bin", two_rooms_two_people())
    _write(FIXTURES_DIR / "hallway_bathroom_door.bin", hallway_bathroom_door())
    _write(FIXTURES_DIR / "single_camera_turn.bin", single_camera_turn())
    _write(FIXTURES_DIR / "cross_camera_handoff.bin", cross_camera_handoff())
    _write(FIXTURES_DIR / "two_people_one_room.bin", two_people_one_room())
    _write(FIXTURES_DIR / "resident_plus_stranger.bin", resident_plus_stranger())
    print("done")


if __name__ == "__main__":
    sys.exit(main())
