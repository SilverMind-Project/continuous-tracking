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
    bbox                dict  x_min/y_min/x_max/y_max

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
) -> dict:
    import uuid as _uuid

    return {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "captured_at_iso": (BASE_TIME + timedelta(seconds=step * FRAME_INTERVAL_S)).isoformat(),
        "floor_x_mm": floor_x_mm,
        "floor_y_mm": floor_y_mm,
        "embedding": embedding,
        "detection_confidence": 0.92,
        "bbox": {"x_min": 100, "y_min": 100, "x_max": 300, "y_max": 400},
        "detection_id": str(_uuid.uuid4()),
        "quality": 0.5,
    }


def _write(path: Path, frames: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for observations in frames:
            data = json.dumps(observations).encode()
            f.write(struct.pack(">I", len(data)))
            f.write(data)
    size_kb = path.stat().st_size // 1024
    print(f"wrote {path.name}: {len(frames)} steps, {size_kb} KB")


def two_cameras_one_room() -> list[list[dict]]:
    """One person walking under two overlapping cameras in one room.

    Steps 0-9:  cam-1 only, person at (3, 5) drifting to (5, 5).
    Steps 10-19: both cameras simultaneously, person at (5, 5) to (7, 5).
    Steps 20-29: cam-2 only, person at (7, 5) to (9, 5).

    The simultaneous-coverage phase (steps 10-19) is the hard case for
    C1: both cameras see the same floor point in the same frame.  The
    WorldTracker's 1-to-1 Hungarian assignment cannot resolve two
    simultaneous observations from distinct cameras to a single PH without
    an upstream cross-camera dedup pass.  T1/T2 are therefore xfail(strict=True)
    — the fixture surfaces the architectural gap; the follow-up (outside R1
    scope) is to implement observation dedup before the assignment step.

    Embedding cluster is [0.9, 0.1, 0.0] ± small noise (same identity).
    """
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
    """Two people in two non-overlapping rooms that never merge.

    Room A: cam-1, person A at (2, 2). Embedding [1.0, 0.0, 0.0].
    Room B: cam-2, person B at (15, 15). Embedding [0.0, 1.0, 0.0].
    Both visible for 20 steps, moving slightly.
    """
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
    """One senior at a bathroom door seen by hallway + doorway camera.

    U1 senior-safety proof: the hallway camera and a doorway camera both observe
    one person standing at the bathroom door.  The cross-camera dedup pass (U1)
    must resolve them to a single Person Hypothesis.

    Steps 0-9:   hallway camera (cam-hall) only; person drifting toward door.
    Steps 10-19: both cameras simultaneously; person at the door (hard case).
    Steps 20-49: camera-blind interval (person inside bathroom, ~15 seconds at
                 0.5 s/frame ≈ 15 s; use 30 steps to represent ~15 minutes at
                 30-second intervals, staying within a compact fixture).
    Steps 50-59: hallway camera (cam-hall) only; person leaving bathroom.

    Embedding cluster is [0.85, 0.15, 0.0] ± small noise (same identity).
    """
    emb_hall = [0.85, 0.15, 0.00]
    emb_door = [0.83, 0.17, 0.00]
    frames: list[list[dict]] = []

    # Steps 0-9: hallway camera only; person walking toward bathroom door.
    for i in range(10):
        fx = 3000 + i * 150  # x_mm 3000-4350
        frames.append([_obs("cam-hall", i, i, fx, 8000, emb_hall)])

    # Steps 10-19: both cameras simultaneously; person at bathroom door.
    for i in range(10):
        step = 10 + i
        fx = 4500 + i * 20  # x_mm 4500-4680
        frames.append(
            [
                _obs("cam-hall", step, step, fx, 8000, emb_hall),
                _obs("cam-door", step, step, fx + 30, 8050, emb_door),
            ]
        )

    # Steps 20-27: camera-blind interval (no observations — person inside bathroom).
    # 8 steps x 0.5 s = 4 s < ph_close_grace_s (5 s), so the PH stays open.
    for _ in range(8):
        frames.append([])

    # Steps 28-37: hallway camera only; person leaving bathroom.
    for i in range(10):
        step = 28 + i
        fx = 4700 - i * 150  # drift back
        frames.append([_obs("cam-hall", step, step, fx, 8000, emb_hall)])

    return frames


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _write(FIXTURES_DIR / "two_cameras_one_room.bin", two_cameras_one_room())
    _write(FIXTURES_DIR / "two_rooms_two_people.bin", two_rooms_two_people())
    _write(FIXTURES_DIR / "hallway_bathroom_door.bin", hallway_bathroom_door())
    print("done")


if __name__ == "__main__":
    sys.exit(main())
