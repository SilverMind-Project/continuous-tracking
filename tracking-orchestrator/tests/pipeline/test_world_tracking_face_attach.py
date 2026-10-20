"""WorldTrackingStage attaches a face anchor only to the observation of its own detection.

Regression proof for the camera-wide fallback: with two people on one camera and a face for
person A only, person B's observation used to inherit A's anchor, and with it A's identity in
the tracker's association identity vectors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

from app.domain import BoundingBox, Detection, FaceAnchor, FloorPoint, WorldObservation
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.tracking.world.tracker import WorldTrackerResult, _unpack_observations
from app.transport.redis_streams import FrameReady

_NOW = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


def _detection(detection_id: str, x_min: int) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id="cam-1",
        bbox=BoundingBox(x_min=x_min, y_min=50, x_max=x_min + 120, y_max=400),
        confidence=0.9,
        floor_point=FloorPoint(1000 + x_min, 2000, calibrated=True),
        embedding=[],
        capture_time=_NOW,
        event_time=_NOW,
    )


def _anchor(person_id: str, detection_id: str, similarity: float) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=similarity,
        similarity=similarity,
        detection_id=detection_id,
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
    )


def _ctx(face_anchors: list[FaceAnchor]) -> FrameContext:
    frame = FrameReady(
        camera_id="cam-1",
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(_NOW.timestamp() * 1e9),
        received_time_unix_ns=int(_NOW.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=_NOW,
        capture_time=_NOW,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = [_detection("det-a", 50), _detection("det-b", 400)]
    ctx.face_anchors = face_anchors
    return ctx


async def _observations_seen_by_tracker(face_anchors: list[FaceAnchor]) -> list[WorldObservation]:
    captured: list[list[WorldObservation]] = []

    async def _step(*, observations: list[WorldObservation], **_kw: Any) -> WorldTrackerResult:
        captured.append(observations)
        return WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])

    tracker = MagicMock()
    tracker.step = AsyncMock(side_effect=_step)
    stage = WorldTrackingStage(
        tracker=tracker,
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        cc_assertion_mode="off",
    )
    await stage.run_many([_ctx(face_anchors)])
    [observations] = captured
    return observations


def _by_detection(observations: list[WorldObservation]) -> dict[str, WorldObservation]:
    return {obs.detection_id: obs for obs in observations}


@pytest.mark.asyncio
async def test_unfaced_person_does_not_inherit_another_persons_face() -> None:
    observations = await _observations_seen_by_tracker([_anchor("alice", "det-a", 0.8)])
    by_det = _by_detection(observations)

    assert by_det["det-a"].face_anchor is not None
    assert by_det["det-a"].face_anchor.person_id == "alice"
    assert by_det["det-b"].face_anchor is None

    vectors = _unpack_observations(observations)
    ids = dict(zip([o.detection_id for o in observations], vectors.face_person_ids, strict=True))
    assert ids == {"det-a": "alice", "det-b": None}


@pytest.mark.asyncio
async def test_anchor_without_detection_id_attaches_to_nothing() -> None:
    orphan = FaceAnchor(
        person_id="alice",
        confidence=0.9,
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
    )
    observations = await _observations_seen_by_tracker([orphan])
    assert all(obs.face_anchor is None for obs in observations)


@pytest.mark.asyncio
async def test_duplicate_anchors_on_one_detection_keep_the_stronger_match() -> None:
    before = REGISTRY.get_sample_value("cts_face_anchor_duplicate_detection_total") or 0.0

    observations = await _observations_seen_by_tracker(
        [_anchor("bob", "det-a", 0.55), _anchor("alice", "det-a", 0.82)]
    )

    face = _by_detection(observations)["det-a"].face_anchor
    assert face is not None
    assert face.person_id == "alice"
    after = REGISTRY.get_sample_value("cts_face_anchor_duplicate_detection_total") or 0.0
    assert after - before == 1
