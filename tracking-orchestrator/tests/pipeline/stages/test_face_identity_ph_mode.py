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
    def __init__(
        self,
        person_id: str,
        confidence: float,
        calibrated_confidence: float | None = None,
        calibration_status: str = "degraded_missing",
        arcface_model_version: str = "buffalo_l",
        preprocessing_version: str = "v1",
        recognition_state: str = "recognized",
    ):
        self.person_id = person_id
        self.confidence = confidence
        self.recognition_state = recognition_state
        self.best_candidate_id = person_id if person_id != "unknown" else None
        self.raw_similarity = confidence
        self.similarity = confidence
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.det_score = 0.85
        self.calibrated_confidence = calibrated_confidence
        self.calibration_status = calibration_status
        self.arcface_model_version = arcface_model_version
        self.preprocessing_version = preprocessing_version


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


# ---------------------------------------------------------------------------
# M10: calibration authority contract tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_similarity_0_95_with_null_calibrated_cannot_satisfy_authority():
    """High raw similarity with degraded calibration must NOT produce a non-None
    calibrated_confidence on the FaceAnchor. The authority gate reads calibrated_confidence
    and fails closed on None.
    """
    client = AsyncMock(spec=FaceIdentificationClient)
    # High raw similarity (0.95) but degraded calibration (status=degraded_missing)
    result = _FakeFaceResult(
        "alice",
        confidence=0.95,
        calibrated_confidence=None,
        calibration_status="degraded_missing",
    )
    client.identify_crops.return_value = [(0, [result])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
        expected_arcface_model_version="buffalo_l",
        expected_preprocessing_version="v1",
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-authority-1")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx.face_anchors, "Expected a FaceAnchor"
    anchor = ctx.face_anchors[0]
    # calibrated_confidence must be None — raw similarity cannot substitute
    assert anchor.calibrated_confidence is None, (
        f"Expected None but got {anchor.calibrated_confidence}; "
        "raw similarity must not satisfy the authority gate"
    )


@pytest.mark.asyncio
async def test_calibrated_confidence_0_85_propagates_to_anchor():
    """calibrated_confidence=0.85 from a ready artifact with matching versions
    must propagate through to FaceAnchor so the resolver can grant authority.
    """
    client = AsyncMock(spec=FaceIdentificationClient)
    result = _FakeFaceResult(
        "alice",
        confidence=0.90,
        calibrated_confidence=0.85,
        calibration_status="ready",
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
    )
    client.identify_crops.return_value = [(0, [result])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
        expected_arcface_model_version="buffalo_l",
        expected_preprocessing_version="v1",
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-authority-2")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx.face_anchors
    anchor = ctx.face_anchors[0]
    assert anchor.calibrated_confidence == pytest.approx(0.85), (
        f"Expected 0.85 but got {anchor.calibrated_confidence}"
    )


@pytest.mark.asyncio
async def test_version_mismatch_nullifies_calibrated_confidence():
    """When arcface_model_version from response doesn't match CTS expectation,
    calibrated_confidence must be set to None on the FaceAnchor.
    """
    client = AsyncMock(spec=FaceIdentificationClient)
    result = _FakeFaceResult(
        "alice",
        confidence=0.90,
        calibrated_confidence=0.88,
        calibration_status="ready",
        arcface_model_version="some_other_model",  # mismatch
        preprocessing_version="v1",
    )
    client.identify_crops.return_value = [(0, [result])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
        expected_arcface_model_version="buffalo_l",
        expected_preprocessing_version="v1",
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-version-mismatch")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx.face_anchors
    anchor = ctx.face_anchors[0]
    assert anchor.calibrated_confidence is None, (
        f"Version mismatch must nullify calibrated_confidence, got {anchor.calibrated_confidence}"
    )


@pytest.mark.asyncio
async def test_calibrated_confidence_propagates_to_face_evidence():
    """calibrated_confidence on FaceAnchor must flow into FaceEvidence."""
    client = AsyncMock(spec=FaceIdentificationClient)
    result = _FakeFaceResult(
        "alice",
        confidence=0.90,
        calibrated_confidence=0.82,
        calibration_status="ready",
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
    )
    client.identify_crops.return_value = [(0, [result])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
        expected_arcface_model_version="buffalo_l",
        expected_preprocessing_version="v1",
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-evidence")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx._face_evidence is not None
    fe = ctx._face_evidence[0]
    assert fe.calibrated_confidence == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_candidate_anchor_never_carries_calibrated_confidence():
    """Candidate state anchors must always have calibrated_confidence=None."""
    client = AsyncMock(spec=FaceIdentificationClient)
    result = _FakeFaceResult(
        "alice",
        confidence=0.35,
        calibrated_confidence=0.50,  # would be valid if recognized
        calibration_status="ready",
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        recognition_state="candidate",
    )
    result.best_candidate_id = "alice"
    client.identify_crops.return_value = [(0, [result])]

    stage = FaceIdentityStage(
        face_id_client=client,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
        expected_arcface_model_version="buffalo_l",
        expected_preprocessing_version="v1",
    )

    ctx = _make_ctx()
    ctx.domain_detections = [_make_detection("det-candidate")]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]

    await stage.run(ctx)
    assert ctx.face_anchors
    anchor = ctx.face_anchors[0]
    assert anchor.recognition_state == "candidate"
    assert anchor.calibrated_confidence is None, (
        "Candidate anchors must never carry calibrated_confidence"
    )
