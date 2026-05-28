"""WTR2: FaceIdentityStage PH-mode tests.

In PH mode (tracklet_manager=None), the stage must:
- Produce FaceAnchors keyed by detection_id.
- Respect cooldown per detection_id.
- Drop low-confidence face results.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FloorPoint
from app.inference.evidence import FaceEvidence
from app.inference.face_id_client import FaceIdentificationClient
from app.observability import metrics as _metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.types import FaceIdCameraConfig


def _make_detection(detection_id: str, camera_id: str = "cam-1") -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id=camera_id,
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
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
    )


class _FakeFaceResult:
    def __init__(self, person_id: str, confidence: float):
        self.person_id = person_id
        self.confidence = confidence


@pytest.mark.asyncio
async def test_ph_mode_emits_face_anchor_with_detection_id():
    """With no tracklet_manager, face ID still produces FaceAnchors with detection_id."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [
        (0, [_FakeFaceResult("alice", 0.85)]),
    ]

    stage = FaceIdentityStage(
        face_id_client=client,
        tracklet_manager=None,  # PH mode
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    det = _make_detection("det-1")
    ctx.domain_detections = [det]
    # Create a minimal (1,1) crop — shape is HWC, np.uint8, non-empty.
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    ctx.effective_width = 640
    ctx.effective_height = 480

    await stage.run(ctx)

    assert len(ctx.face_anchors) == 1
    fa = ctx.face_anchors[0]
    assert fa.person_id == "alice"
    assert fa.detection_id == "det-1"
    assert fa.tracklet_id == ""  # no tracklet manager
    assert fa.confidence == 0.85


@pytest.mark.asyncio
async def test_ph_mode_cooldown_keyed_by_detection_id():
    """Cooldown in PH mode is keyed by detection_id, not tracklet_id."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [
        (0, [_FakeFaceResult("alice", 0.85)]),
    ]

    stage = FaceIdentityStage(
        face_id_client=client,
        tracklet_manager=None,
        face_id_cooldown_s=5.0,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    det = _make_detection("det-1")
    ctx.domain_detections = [det]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    ctx.effective_width = 640
    ctx.effective_height = 480

    # First call — produces anchor.
    await stage.run(ctx)
    assert len(ctx.face_anchors) == 1

    # Second call with same detection_id — should skip due to cooldown.
    ctx2 = _make_ctx()
    det2 = _make_detection("det-1")  # same detection_id
    ctx2.domain_detections = [det2]
    ctx2.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    ctx2.effective_width = 640
    ctx2.effective_height = 480

    await stage.run(ctx2)
    assert len(ctx2.face_anchors) == 0  # cooldown skip


@pytest.mark.asyncio
async def test_low_confidence_face_dropped():
    """Face results below min_confidence must not produce anchors."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [
        (0, [_FakeFaceResult("alice", 0.35)]),
    ]

    stage = FaceIdentityStage(
        face_id_client=client,
        tracklet_manager=None,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    det = _make_detection("det-1")
    ctx.domain_detections = [det]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    ctx.effective_width = 640
    ctx.effective_height = 480

    await stage.run(ctx)
    assert len(ctx.face_anchors) == 0


@pytest.mark.asyncio
async def test_face_evidence_includes_detection_id_in_ph_mode():
    """FaceEvidence carries detection_id in PH mode."""
    client = AsyncMock(spec=FaceIdentificationClient)
    client.identify_crops.return_value = [
        (0, [_FakeFaceResult("alice", 0.85)]),
    ]

    stage = FaceIdentityStage(
        face_id_client=client,
        tracklet_manager=None,
        face_id_min_confidence=0.5,
        face_id_camera_configs={"cam-1": FaceIdCameraConfig(enabled=True)},
    )

    ctx = _make_ctx()
    det = _make_detection("det-1")
    ctx.domain_detections = [det]
    ctx.crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    ctx.effective_width = 640
    ctx.effective_height = 480

    await stage.run(ctx)
    assert ctx._face_evidence is not None
    assert len(ctx._face_evidence) == 1
    fe = ctx._face_evidence[0]
    assert fe.detection_id == "det-1"
    assert fe.person_id == "alice"
