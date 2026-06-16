"""Shared replay fixture loader for WorldTracker integration tests.

Provides :func:`load_fixture` for reading length-prefixed JSON .bin fixture
files and constructing :class:`WorldObservation` lists.  Also provides
``FIXTURES_DIR``, ``_ROOM_POLYGONS``, and :func:`load_truth` for reuse.

Extended to support ``calibrated`` (default True for backward
compatibility with existing fixtures) and ``face_anchor`` fields.
Extended to support ``orientation`` and ``orientation_confidence``.
Extended to support ``floor_cov_random`` and ``footpoint_reliable``.
"""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain import BoundingBox, FaceAnchor, FloorPoint, OrientationBin, WorldObservation

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frame_replays"

# A large room polygon that covers all fixture floor coordinates.
# Fixture ranges: x_mm 3000-11000 (3-11 m), y_mm 2000-16000 (2-16 m).
_ROOM_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "living_room": [(0.0, 0.0), (25.0, 0.0), (25.0, 25.0), (0.0, 25.0)]
}


def load_fixture(path: Path) -> list[list[WorldObservation]]:
    """Load length-prefixed JSON fixture; return list of per-frame observation lists.

    Format: 4-byte big-endian uint32 length prefix, followed by that many
    bytes of UTF-8 JSON.  Each JSON payload is a list of observation dicts
    (one per observation in that frame step).
    """
    steps: list[list[WorldObservation]] = []
    with path.open("rb") as f:
        while chunk := f.read(4):
            length = struct.unpack(">I", chunk)[0]
            data = f.read(length)
            obs_list: list[dict[str, Any]] = json.loads(data)
            frame_obs: list[WorldObservation] = []
            for o in obs_list:
                bbox_d = o["bbox"]

                # ── face anchor (optional) ──
                face_anchor: FaceAnchor | None = None
                fa_data = o.get("face_anchor")
                if fa_data is not None:
                    face_anchor = FaceAnchor(
                        person_id=fa_data["person_id"],
                        confidence=float(fa_data["confidence"]),
                        quality=float(fa_data.get("quality", 1.0)),
                        detection_id=fa_data.get("detection_id", ""),
                        camera_id=fa_data.get("camera_id", ""),
                        captured_at=(
                            datetime.fromisoformat(fa_data["captured_at_iso"])
                            if fa_data.get("captured_at_iso")
                            else datetime(2026, 1, 1, tzinfo=UTC)
                        ),
                    )

                # ── orientation (optional) ──
                ori_raw = o.get("orientation", OrientationBin.UNKNOWN)
                try:
                    orientation = OrientationBin(int(ori_raw))
                except (ValueError, KeyError):
                    orientation = OrientationBin.UNKNOWN
                orientation_confidence = float(o.get("orientation_confidence", 0.0))
                floor_cov_raw = o.get("floor_cov_random")
                floor_cov_random = (
                    (
                        float(floor_cov_raw[0]),
                        float(floor_cov_raw[1]),
                        float(floor_cov_raw[2]),
                        float(floor_cov_raw[3]),
                    )
                    if floor_cov_raw is not None
                    else None
                )

                frame_obs.append(
                    WorldObservation(
                        camera_id=o["camera_id"],
                        frame_index=o["frame_index"],
                        captured_at=datetime.fromisoformat(o["captured_at_iso"]),
                        floor_point=FloorPoint(
                            x_mm=o["floor_x_mm"],
                            y_mm=o["floor_y_mm"],
                            calibrated=o.get("calibrated", True),
                        ),
                        bbox=BoundingBox(
                            x_min=bbox_d["x_min"],
                            y_min=bbox_d["y_min"],
                            x_max=bbox_d["x_max"],
                            y_max=bbox_d["y_max"],
                        ),
                        embedding=o["embedding"],
                        detection_confidence=o["detection_confidence"],
                        detection_id=o.get("detection_id", ""),
                        quality=float(o.get("quality", 0.5)),
                        observation_id=o.get("observation_id", ""),
                        height_estimate_m=o.get("height_estimate_m"),
                        floor_residual_m=o.get("floor_residual_m"),
                        face_anchor=face_anchor,
                        orientation=orientation,
                        orientation_confidence=orientation_confidence,
                        floor_cov_random=floor_cov_random,
                        footpoint_reliable=bool(o.get("footpoint_reliable", True)),
                    )
                )
            steps.append(frame_obs)
    return steps


def load_truth(path: Path) -> dict:
    """Load a truth sidecar JSON alongside a fixture binary.

    Returns a dict with keys:
        persons:          list[str]
        detection_truth:  dict[str, str]  detection_id → person label
        events:           list[dict]
    """
    truth_path = path.with_suffix(".truth.json")
    if not truth_path.exists():
        return {"persons": [], "detection_truth": {}, "events": []}
    with truth_path.open() as f:
        return json.load(f)
