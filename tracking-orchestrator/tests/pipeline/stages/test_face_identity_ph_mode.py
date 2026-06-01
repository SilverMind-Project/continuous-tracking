"""FaceIdentityStage PH-mode tests.

The stage must:
- Produce FaceAnchors keyed by detection_id.
- Suppress repeat calls via per-track IoU-based cooldown.
- Force face-id for both detections when their bboxes overlap (crossing).
- Prune cooldown entries for tracks that disappear from the frame.
- Drop low-confidence face results.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FloorPoint
from app.inference.face_id_client import FaceIdentificationClient
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import (
    FaceIdentityStage,
    _bbox_iou,
    _crossing_indices,
    _match_to_tracks,
    _TrackEntry,
)
from app.pipeline.types import FaceIdCameraConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bbox(x_min: int, y_min: int, x_max: int, y_max: int) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _make_detection(
    detection_id: str,
    camera_id: str = "cam-1",
    bbox: BoundingBox | None = None,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id=camera_id,
        bbox=bbox or _make_bbox(10, 20, 110, 220),
        embedding=[],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=0.9,
        floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
    )


def _make_ctx(camera_id: str = "cam-1") -> FrameContext:
    from app.transport.redis_streams import FrameReady

    frame = FrameReady(
        camera_id=camera_id,
        minio_key="test/key",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
    )
    return FrameContext(
        frame=frame,
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
        effective_width=640,
        effective_height=480,
    )


class _FakeFaceResult:
    def __init__(self, person_id: str, confidence: float):
        self.person_id = person_id
        self.confidence = confidence
        self.recognition_state = "recognized"
        self.best_candidate_id = person_id if person_id != "unknown" else None
        self.similarity = confidence
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.det_score = 0.85


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_bbox_iou_identical():
    bbox = _make_bbox(0, 0, 100, 100)
    assert _bbox_iou(bbox, bbox) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    a = _make_bbox(0, 0, 50, 50)
    b = _make_bbox(100, 100, 200, 200)
    assert _bbox_iou(a, b) == 0.0


def test_bbox_iou_partial():
    a = _make_bbox(0, 0, 100, 100)
    b = _make_bbox(50, 0, 150, 100)  # 50-wide overlap
    iou = _bbox_iou(a, b)
    assert 0.0 < iou < 1.0


def test_match_to_tracks_inherits_id_on_high_iou():
    det = _make_detection("d1", bbox=_make_bbox(10, 20, 110, 220))
    prev = [_TrackEntry(track_id="stable-id", bbox=_make_bbox(10, 20, 110, 220))]
    ids = _match_to_tracks([det], prev)
    assert ids[0] == "stable-id"


def test_match_to_tracks_fresh_id_on_low_iou():
    det = _make_detection("d1", bbox=_make_bbox(500, 400, 600, 480))
    prev = [_TrackEntry(track_id="old-id", bbox=_make_bbox(10, 20, 110, 220))]
    ids = _match_to_tracks([det], prev)
    assert ids[0] != "old-id"


def test_match_to_tracks_no_double_assignment():
    bbox = _make_bbox(10, 20, 110, 220)
    dets = [_make_detection("d1", bbox=bbox), _make_detection("d2", bbox=bbox)]
    prev = [_TrackEntry(track_id="track-A", bbox=bbox)]
    ids = _match_to_tracks(dets, prev)
    # Only one detection can inherit; the second gets a fresh ID.
    assert ids.count("track-A") == 1
    assert ids[0] != ids[1]


def test_crossing_indices_overlap_flags_both():
    a = _make_detection("d1", bbox=_make_bbox(0, 0, 200, 400))
    b = _make_detection("d2", bbox=_make_bbox(100, 0, 300, 400))  # large overlap
    result = _crossing_indices([a, b])
    assert 0 in result
    assert 1 in result


def test_crossing_indices_no_overlap():
    a = _make_detection("d1", bbox=_make_bbox(0, 0, 100, 200))
    b = _make_detection("d2", bbox=_make_bbox(300, 0, 400, 200))
    assert len(_crossing_indices([a, b])) == 0


def test_crossing_indices_three_people_only_overlapping_flagged():
    # a and b overlap heavily (x overlap = 120px on 200px-wide boxes → IoU ≈ 0.43).
    a = _make_detection("d1", bbox=_make_bbox(0, 0, 200, 400))
    b = _make_detection("d2", bbox=_make_bbox(80, 0, 280, 400))  # crosses a
    c = _make_detection("d3", bbox=_make_bbox(500, 0, 600, 400))  # isolated
    result = _crossing_indices([a, b, c])
    assert 0 in result
    assert 1 in result
    assert 2 not in result


# ---------------------------------------------------------------------------
# Integration tests: FaceIdentityStage behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ph_mode_emits_face_anchor_with_detection_id():
    """Face ID produces FaceAnchors keyed by detection_id."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [
        (0, [_FakeFaceResult("alice", 0.85)]),
    ]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-1")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)

    assert len(ctx.face_anchors) == 1
    fa = ctx.face_anchors[0]
    assert fa.person_id == "alice"
    assert fa.detection_id == "det-1"
    assert fa.tracklet_id == ""
    assert fa.confidence == 0.85


@pytest.mark.asyncio
async def test_cooldown_fires_when_track_position_unchanged():
    """Same bbox between frames → same track ID → cooldown suppresses second call."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [(0, [_FakeFaceResult("alice", 0.85)])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_cooldown_s=5.0,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    bbox = _make_bbox(10, 20, 110, 220)

    ctx1 = _make_ctx()
    ctx1.domain_detections = [_make_detection("det-1", bbox=bbox)]
    ctx1.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    await stage.run(ctx1)
    assert len(ctx1.face_anchors) == 1

    # Second frame: same bbox → high IoU → same track ID → cooldown fires.
    ctx2 = _make_ctx()
    ctx2.domain_detections = [_make_detection("det-2", bbox=bbox)]  # fresh detection_id
    ctx2.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    await stage.run(ctx2)

    assert len(ctx2.face_anchors) == 0
    assert client.identify_crops.call_count == 1  # only called once


@pytest.mark.asyncio
async def test_new_position_bypasses_cooldown():
    """Detection with no IoU match (new person or large movement) bypasses cooldown."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [(0, [_FakeFaceResult("bob", 0.85)])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_cooldown_s=60.0,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    # Seed a track at position A.
    ctx1 = _make_ctx()
    ctx1.domain_detections = [_make_detection("det-1", bbox=_make_bbox(0, 0, 100, 200))]
    ctx1.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    await stage.run(ctx1)
    assert client.identify_crops.call_count == 1

    # Second frame: completely different position → new track → face-id called again.
    ctx2 = _make_ctx()
    ctx2.domain_detections = [_make_detection("det-2", bbox=_make_bbox(500, 300, 600, 480))]
    ctx2.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    await stage.run(ctx2)
    assert client.identify_crops.call_count == 2


@pytest.mark.asyncio
async def test_crossing_forces_face_id_despite_cooldown():
    """Two overlapping detections bypass the cooldown so identities stay consistent."""
    call_count = 0

    async def _mock_identify(crops, bboxes):
        nonlocal call_count
        call_count += 1
        return [(i, [_FakeFaceResult(f"person-{i}", 0.85)]) for i in range(len(crops))]

    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.side_effect = _mock_identify

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_cooldown_s=60.0,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    bbox_a = _make_bbox(0, 0, 200, 400)
    bbox_b = _make_bbox(300, 0, 500, 400)

    # Frame 1: two separated people — seed their tracks.
    ctx1 = _make_ctx()
    ctx1.domain_detections = [
        _make_detection("d1", bbox=bbox_a),
        _make_detection("d2", bbox=bbox_b),
    ]
    ctx1.crops = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2
    await stage.run(ctx1)
    assert call_count == 1  # both called in one batch

    # Frame 2: bboxes overlap — crossing detected → cooldown bypassed.
    bbox_cross_a = _make_bbox(100, 0, 300, 400)  # overlapping
    bbox_cross_b = _make_bbox(150, 0, 350, 400)  # overlapping
    ctx2 = _make_ctx()
    ctx2.domain_detections = [
        _make_detection("d3", bbox=bbox_cross_a),
        _make_detection("d4", bbox=bbox_cross_b),
    ]
    ctx2.crops = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2
    await stage.run(ctx2)
    assert call_count == 2  # called again despite 60-second cooldown


@pytest.mark.asyncio
async def test_disappeared_track_pruned_from_cooldown_dict():
    """Tracks that vanish from the frame are removed from _last_face_id_by_tracklet."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [(0, [_FakeFaceResult("alice", 0.85)])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_cooldown_s=60.0,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    # Frame 1: person visible — seeds track and cooldown entry.
    ctx1 = _make_ctx()
    ctx1.domain_detections = [_make_detection("det-1")]
    ctx1.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    await stage.run(ctx1)
    assert len(stage._last_face_id_by_tracklet) == 1

    # Frame 2: no detections — person left the frame.
    ctx2 = _make_ctx()
    ctx2.domain_detections = []
    ctx2.crops = []
    await stage.run(ctx2)

    assert len(stage._last_face_id_by_tracklet) == 0  # entry pruned


@pytest.mark.asyncio
async def test_low_confidence_face_dropped():
    """Face results below min_confidence must not produce anchors."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [(0, [_FakeFaceResult("alice", 0.35)])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-1")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert len(ctx.face_anchors) == 0


@pytest.mark.asyncio
async def test_face_evidence_includes_detection_id_in_ph_mode():
    """FaceEvidence carries detection_id in PH mode."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [(0, [_FakeFaceResult("alice", 0.85)])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-1")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx._face_evidence is not None
    assert len(ctx._face_evidence) == 1
    fe = ctx._face_evidence[0]
    assert fe.detection_id == "det-1"
    assert fe.person_id == "alice"
